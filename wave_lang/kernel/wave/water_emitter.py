#!/usr/bin/env python3
"""
Water Emitter for Wave Dialect

This generates operations Wave dialect from a serialized wave trace (json).
It runs as a standalone process with access to Water Python bindings and emits
Wave operations with fallback support for unknown operations.
"""

import json
import sys
import sympy
from functools import reduce

try:
    from water_mlir import ir
    from water_mlir.water_mlir.dialects.wave import (
        AddOp,
        DivOp,
        Exp2Op,
        MulOp,
        ReadOp,
        RegisterOp,
        WriteOp,
        WaveSymbolAttr,
        WaveIndexMappingAttr,
    )
    from water_mlir.water_mlir.dialects import wave
except Exception as e:
    print(f"FATAL: failed to import water_mlir: {e}", file=sys.stderr)
    sys.exit(2)

# Mapping from tkw_op_name to actual op constructors
WAVE_OP_CONSTRUCTORS = {
    "add": AddOp,
    "mul": MulOp,
    "div": DivOp,
    "exp2": Exp2Op,
    "read": ReadOp,
    "write": WriteOp,
    "register": RegisterOp,
    # TODO: Add more or find a good way of avoiding needing a mapping
}


def _default_tensor_type():
    # Return a default Wave tensor type for register operations
    return ir.Type.parse("!wave.tensor<any of f32, <register>>")


def _default_memory_tensor_type():
    # Return a default Wave tensor type for memory operations
    return ir.Type.parse("!wave.tensor<any of f32, <global>>")


def _map_address_space(addr: str) -> str:
    if not addr:
        return ""
    a = str(addr).lower()
    if "global" in a:
        return "global"
    if "shared" in a:
        return "shared"
    if "register" in a:
        return "register"
    # If we can't determine the address space, use unspecified
    return "unspecified"


def _build_wave_type_text(wave_type_dict: dict) -> str:
    kind = wave_type_dict.get("kind")
    shape = wave_type_dict.get("shape", []) or []
    dtype = wave_type_dict.get("dtype", "f32")
    addr = wave_type_dict.get("address_space", "")
    if not shape:
        shape_txt = "any"
    else:
        shape_txt = ", ".join([f"@{s}" for s in shape])
    addr_norm = _map_address_space(addr)
    addr_txt = f", <{addr_norm}>" if addr_norm else ""
    return f"!wave.tensor<[{shape_txt}] of {dtype}{addr_txt}>"


def _type_from_wave_type_dict(ctx: ir.Context, wave_type_dict: dict) -> ir.Type:
    # Use textual parser to construct the Wave type.
    type_text = _build_wave_type_text(wave_type_dict)
    return ir.Type.parse(type_text, context=ctx)


def sympy_to_affine_map(expr, symbol_names):
    """
    Convert a sympy expression to an MLIR AffineMap.

    Args:
        expr: sympy expression
        symbol_names: list of symbol names in order (matches s0, s1, s2, ...)

    Returns:
        AffineMap representing the expression
    """

    def convert_expr(sympy_expr):
        """Recursively convert sympy expression to AffineExpr"""
        if sympy_expr.is_Integer:
            return ir.AffineExpr.get_constant(sympy_expr.p)

        elif sympy_expr.is_Symbol:
            # Find the index of this symbol in our symbol_names list
            symbol_name = str(sympy_expr).strip()
            if symbol_name in symbol_names:
                symbol_idx = symbol_names.index(symbol_name)
                return ir.AffineExpr.get_symbol(symbol_idx)
            else:
                raise ValueError(f"Unknown symbol: {symbol_name}")

        elif sympy_expr.is_Add:
            # Convert addition: a + b + c
            result = ir.AffineExpr.get_constant(0)
            for term in sympy_expr.args:
                expr = convert_expr(term)
                result = result + expr
            return result

        elif sympy_expr.is_Mul:
            # Convert multiplication: a * b * c
            result = ir.AffineExpr.get_constant(1)
            divide_by = 1
            for factor in sympy_expr.args:
                if factor.is_Rational:
                    assert factor.p == 1
                    divide_by *= factor.q
                    continue
                if divide_by > 1:
                    expr = convert_expr(sympy.floor(factor / divide_by))
                    result *= expr
                    divide_by = 1
                    continue
                result *= convert_expr(factor)

            return result

        elif hasattr(sympy_expr, 'func'):
            # Handle special functions
            func_name = sympy_expr.func.__name__

            if func_name == 'floor' and len(sympy_expr.args) == 1:
                print("CONERTING FLOOR ", sympy_expr, sympy_expr.args[0].is_Mul, len(sympy_expr.args[0].args) == 2)
                if sympy_expr.args[0].is_Mul and len(sympy_expr.args[0].args) == 2: #and sympy_expr.args[0].args[0].is_Rational:
                    print("CONERTING FLOOR ", sympy_expr.args[0].args[0], type(sympy_expr.args[0].args[0]))
                    print("CONERTING FLOOR ", sympy_expr.args[0].args[1], type(sympy_expr.args[0].args[1]))
                    if sympy_expr.args[0].args[0].is_Rational:
                        numerator = convert_expr(sympy_expr.args[0].args[1])
                        denominator = sympy_expr.args[0].args[0].q
                    else:
                        numerator = convert_expr(sympy_expr.args[0].args[0])
                        denominator = convert_expr(sympy_expr.args[0].args[1].q)
                    print("CONERTING FLOOR ", numerator, denominator)
                    tmp = ir.AffineExpr.get_floor_div(numerator, denominator)
                    print("FLOOR RES ", tmp)
                    return tmp

                arg = convert_expr(sympy_expr.args[0])
                print("CONVERTING FLOOR ", arg)
                return arg.floor_div(ir.AffineExpr.constant(1))

            elif func_name == 'ceiling' and len(sympy_expr.args) == 1:
                print("CONVERTING FLOOR ", sympy_expr.args[0])
                print("CONVERTING CEIL ", len(sympy_expr.args[0].args))
                print("CONVERTING CEIL ", sympy_expr.args[0].args[1].is_Rational)
                if sympy_expr.args[0].is_Mul and len(sympy_expr.args[0].args) == 2 and sympy_expr.args[0].args[0].is_Rational:
                    numerator = convert_expr(sympy_expr.args[0].args[1])
                    tmp = ir.AffineExpr.get_ceil_div(numerator, sympy_expr.args[0].args[0].q)
                    print("CEIL RES ", tmp)
                    return tmp

                # ceiling(x) -> (x + 1 - 1) floordiv 1 (simplified)
                arg = convert_expr(sympy_expr.args[0])
                return (arg + ir.AffineExpr.constant(1) -
                        ir.AffineExpr.constant(1)).floor_div(
                        ir.AffineExpr.constant(1))

            elif func_name == 'Mod' and len(sympy_expr.args) == 2:
                # Mod(x, y) -> x mod y
                x = convert_expr(sympy_expr.args[0])
                y = convert_expr(sympy_expr.args[1])
                return x % y

            else:
                raise ValueError(f"Unsupported function: {func_name}")

        else:
            raise ValueError(f"Unsupported expression type: {sympy_expr}")

    try:
        affine_expr = convert_expr(expr)
        # Create affine map with 0 dimensions and len(symbol_names) symbols
        return ir.AffineMap.get(0, len(symbol_names), [affine_expr])
    except Exception as e:
        raise ValueError(f"Failed to convert expression {expr}: {e}")


def create_operation_attributes(node_data):
    """Create MLIR attributes from node data."""
    attrs = {}

    # Map common attributes
    if "name" in node_data:
        attrs["fx.name"] = ir.StringAttr.get(str(node_data["name"]))
    if "op" in node_data:
        attrs["fx.op"] = ir.StringAttr.get(str(node_data["op"]))
    if "tkw_op_name" in node_data:
        attrs["fx.tkw_op_name"] = ir.StringAttr.get(str(node_data["tkw_op_name"]))
    if "subgraph" in node_data:
        attrs["fx.subgraph"] = ir.StringAttr.get(str(node_data["subgraph"]))

    # Map Python-specific attributes
    for py_attr in (
        "vector_shapes",
        "reduction_dim",
        "iter_idx",
        "location",
        "expanded_dims",
        "scheduling_parameters",
    ):
        if py_attr in node_data:
            attrs[f"py.{py_attr}"] = ir.StringAttr.get(str(node_data[py_attr]))

    # Handle index attribute - store as string for now
    if "index" in node_data:
        index_str = node_data["index"]

        input_string = index_str.strip().strip('{}').strip()

        result = {}
        current_key = None
        current_value = ""
        paren_level = 0
        i = 0
        start_index = 0  # Track the start of the current entry

        while i < len(input_string):
            char = input_string[i]

            # Track parentheses level
            if char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1

            # Comma at top level (paren_level == 0) indicates new entry
            if char == ',' and paren_level == 0:
                if current_key is not None:
                   #result[current_key] = ir.StringAttr.get(current_value.strip())
                    current_value = "$WG0*BLOCK_M + BLOCK_M*floor($T0/64)/4/8/2 + Mod($T0, 64) : floor(BLOCK_M/BLOCK_N) : 64"
                    idx_exprs = current_value.replace('$', '').split(':')
                    sp_exprs = [sympy.parsing.sympy_parser.parse_expr(idx_expr) for idx_expr in idx_exprs]
                    symbol_names = [str(sym) for sym in reduce(lambda x, y: x.union(y.free_symbols), sp_exprs, set())]

                    # Create new symbols with positive assumptions
                    all_symbols = {}
                    for symbol_name in symbol_names:
                        all_symbols[symbol_name] = sympy.Symbol(symbol_name, positive=True)
                    print(all_symbols)

                    # Now re-parse with positive assumptions
                    sp_exprs = [sympy.sympify(idx_expr, locals=all_symbols) for idx_expr in idx_exprs]
                    all_symbols = [str(sym) for sym in all_symbols.values()]

                    print("SYMBOLS ", all_symbols)
                    print("SP EXPrs ", sp_exprs)
                    start_map = sympy_to_affine_map(sp_exprs[0], all_symbols)
                    step_map = sympy_to_affine_map(sp_exprs[1], all_symbols)
                    stride_map = sympy_to_affine_map(sp_exprs[2], all_symbols)
                    print("START ", start_map)
                    print("STEP ", step_map)
                    print("STRIDE ", stride_map)
                    result[current_key] = WaveIndexMappingAttr.get(all_symbols, start_map, step_map, stride_map)
                current_key = None
                current_value = ""
                i += 1
                # Skip whitespace after comma
                while i < len(input_string) and input_string[i].isspace():
                    i += 1
                start_index = i
                continue

            if current_key is None and char == ':':
                # Found key-value separator - extract key from current entry segment
                key_segment = input_string[start_index:i].strip()
                if key_segment:
                    current_key = key_segment
                    current_value = ""
                    # Skip the colon and any whitespace
                    i += 1
                    while i < len(input_string) and input_string[i].isspace():
                        i += 1
                    continue

            if current_key is not None:
                current_value += char

            i += 1

        # Add the last entry
        print("CP")
        if current_key is not None:
           #result[current_key] = ir.StringAttr.get(current_value.strip())
            idx_exprs = current_value.replace('$', '').split(':')
            sp_exprs = [sympy.parsing.sympy_parser.parse_expr(idx_expr) for idx_expr in idx_exprs]
            symbol_names = [str(sym) for sym in reduce(lambda x, y: x.union(y.free_symbols), sp_exprs, set())]

            # Create new symbols with positive assumptions
            all_symbols = {}
            for symbol_name in symbol_names:
                all_symbols[symbol_name] = sympy.Symbol(symbol_name, positive=True)
            print(all_symbols)

            # Now re-parse with positive assumptions
            sp_exprs = [sympy.sympify(idx_expr, locals=all_symbols) for idx_expr in idx_exprs]
            all_symbols = [str(sym) for sym in all_symbols.values()]
            print("SYMBOLS ", all_symbols)
            print("SP EXPrs ", sp_exprs)
            start_map = sympy_to_affine_map(sp_exprs[0], all_symbols)
            step_map = sympy_to_affine_map(sp_exprs[1], all_symbols)
            stride_map = sympy_to_affine_map(sp_exprs[2], all_symbols)
            print("START ", start_map)
            print("STEP ", step_map)
            print("STRIDE ", stride_map)
            result[current_key] = WaveIndexMappingAttr.get(all_symbols, start_map, step_map, stride_map)
        print("RES ", result)

        attrs["py.index"] = ir.DictAttr.get(result)
        print("EMITTER NODE ", node_data)
        print("EMITTER ATTR ", index_str)
        print("EMITTER ATTR ", ir.StringAttr.get(index_str))
        print("EMITTER ATTR ", attrs["py.index"])

    return attrs


def create_unregistered_operation(op_name, node_data, insertion_point):
    """Fallback to create unregistered operation for unknown ops."""
    attrs = create_operation_attributes(node_data)

    return ir.Operation.create(
        name=op_name,
        attributes=attrs,
        results=[],
        operands=[],
    )


def main():
    try:
        data = json.load(sys.stdin)
    except Exception as e:
        print(f"FATAL: failed to read JSON: {e}", file=sys.stderr)
        return 3

    with ir.Context() as ctx:
        # Only allow registered dialects/ops
        try:
            ctx.allow_unregistered_dialects = False
        except Exception:
            pass
        wave.register_dialect(ctx)
        with ir.Location.unknown():
            module = ir.Module.create()
            # Build a function with arguments matching placeholders and optional returns.
            id_to_value = {}
            name_to_value = {}
            func_spec = data.get("function", {}) or {}
            func_args = func_spec.get("args", [])
            func_rets = func_spec.get("returns", [])
            # Derive argument types from corresponding placeholder nodes' wave_type if present.
            # Index func_args by id to find matching node metadata.
            nodes_by_id = {n.get("id"): n for n in data.get("nodes", [])}
            arg_types = []
            for arg in func_args:
                nid = arg.get("id")
                nmeta = nodes_by_id.get(nid, {}) if nid is not None else {}
                wave_type = nmeta.get("wave_type")
                if wave_type:
                    try:
                        arg_types.append(_type_from_wave_type_dict(ctx, wave_type))
                        continue
                    except Exception:
                        pass
                arg_types.append(_default_memory_tensor_type())
            ret_types = [_default_tensor_type() for _ in func_rets]
            func_type = ir.FunctionType.get(arg_types, ret_types)
            with ir.InsertionPoint(module.body):
                func_op = ir.Operation.create(
                    name="func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get("kernel"),
                        "function_type": ir.TypeAttr.get(func_type),
                    },
                    results=[],
                    operands=[],
                    regions=1,
                )
                entry_block = ir.Block.create_at_start(func_op.regions[0], arg_types)
                # Map function arguments by declared ids/names.
                for i, arg in enumerate(func_args):
                    arg_id = arg.get("id")
                    arg_name = arg.get("name", "")
                    if arg_id is not None:
                        id_to_value[arg_id] = entry_block.arguments[i]
                    if arg_name:
                        name_to_value[arg_name] = entry_block.arguments[i]

                # Emit body operations.
                with ir.InsertionPoint(entry_block):
                    for node in data.get("nodes", []):
                        tkw_op_name = node.get("tkw_op_name", "unknown")
                        # Skip ops that will not be materialized
                        if tkw_op_name in ("placeholder", "output"):
                            continue

                        # Try to create the operation using actual constructors or fallback
                        try:
                            # Prefer structured inputs (inputs_ex) falling back to names.
                            operands = []
                            if "inputs_ex" in node:
                                for ref in node.get("inputs_ex", []):
                                    pid = ref.get("id")
                                    ridx = int(ref.get("result", 0) or 0)
                                    val = id_to_value.get(pid)
                                    if val is not None:
                                        operands.append(val if ridx == 0 else None)
                            else:
                                for on in node.get("inputs", []):
                                    v = name_to_value.get(on)
                                    if v is not None:
                                        operands.append(v)

                            # Determine result type for ops that produce a single result.
                            result_type = None
                            if tkw_op_name in {
                                "add",
                                "mul",
                                "div",
                                "exp2",
                                "mma",
                                "read",
                                "register",
                            }:
                                wave_type = node.get("wave_type")
                                if wave_type:
                                    try:
                                        result_type = _type_from_wave_type_dict(
                                            ctx, wave_type
                                        )
                                    except Exception:
                                        # If wave_type parsing fails, we should have inferred the type
                                        # in mlir_converter.py, so this shouldn't happen
                                        result_type = None
                                else:
                                    # No wave_type available, we should have inferred the type
                                    # in mlir_converter.py, so this shouldn't happen
                                    result_type = None

                            attrs = create_operation_attributes(node)

                            # Try to use actual op constructor first
                            op = None
                            if tkw_op_name in WAVE_OP_CONSTRUCTORS:
                                ctor = WAVE_OP_CONSTRUCTORS[tkw_op_name]
                                # TODO: This could be more elegant
                                if (
                                    tkw_op_name in {"add", "mul", "div"}
                                    and len(operands) == 2
                                    and result_type is not None
                                ):
                                    op = ctor(result_type, operands[0], operands[1])
                                elif (
                                    tkw_op_name == "exp2"
                                    and len(operands) == 1
                                    and result_type is not None
                                ):
                                    op = ctor(result_type, operands[0])
                                elif (
                                    tkw_op_name == "read"
                                    and len(operands) == 1
                                    and result_type is not None
                                ):
                                    op = ctor(result_type, operands[0], index=attrs["py.index"])
                                    print("CREATING READ ", op)
                                    op.attributes["index"] = attrs["py.index"]
                                    print("CREATING READ ", attrs["py.index"])
                                    print("CREATING READ ", op.attributes)
                                elif tkw_op_name == "write" and len(operands) == 2:
                                    op = ctor(operands[0], operands[1])
                                    op.attributes["index"] = attrs["py.index"]
                                    print("CREATING WRITE ", op)
                                    op.attributes["index"] = attrs["py.index"]
                                    print("CREATING WRITE ", attrs["py.index"])
                                    print("CREATING WRITE ", op.attributes["index"])
                                elif tkw_op_name == "register":
                                    if result_type is not None:
                                        # wave.register requires a scalar initialization value
                                        # Try to get the constant value from the trace, fallback to 0.0
                                        constant_value = 0.0
                                        if (
                                            "arg_2" in node
                                        ):  # The value is the 3rd argument (index 2) in NewRegister
                                            try:
                                                constant_value = float(node["arg_2"])
                                            except (ValueError, TypeError):
                                                print(
                                                    f"WARNING: Could not parse constant value '{node.get('arg_2', '')}', using 0.0",
                                                    file=sys.stderr,
                                                )

                                        try:
                                            # Try to get element type from wave tensor type
                                            element_type = result_type.element_type
                                        except AttributeError:
                                            # Fallback to f16 if we can't determine element type (matches test)
                                            element_type = ir.F16Type.get()

                                        # Create arith.constant for the scalar value
                                        constant_op = ir.Operation.create(
                                            name="arith.constant",
                                            attributes={
                                                "value": ir.FloatAttr.get(
                                                    element_type, constant_value
                                                )
                                            },
                                            results=[element_type],
                                            operands=[],
                                        )

                                        # Create the register op with the constant as operand
                                        op = ctor(result_type, constant_op.results[0])
                                    else:
                                        # Type inference failed, skip this operation
                                        print(
                                            f"WARNING: Could not infer type for register operation, skipping",
                                            file=sys.stderr,
                                        )
                                        continue

                            # Fallback to generic operation creation for unknown ops or when constructor fails
                            if op is None:
                                op_name = f"wave.{tkw_op_name if tkw_op_name != 'yield' else 'yield'}"
                                result_types = (
                                    [result_type] if result_type is not None else []
                                )
                                op = ir.Operation.create(
                                    name=op_name,
                                    attributes=attrs,
                                    results=result_types,
                                    operands=operands,
                                )

                            res = op.results[0] if len(op.results) > 0 else None
                            # Track by name and by optional node id.
                            if res is not None:
                                name_to_value[node.get("name", "")] = res
                                nid = node.get("id")
                                if nid is not None:
                                    id_to_value[nid] = res

                        except Exception as e:
                            print(
                                f"ERROR: failed to create op '{tkw_op_name}': {e}",
                                file=sys.stderr,
                            )
                    # Emit function return.
                    ret_operands = []
                    for r in func_rets:
                        pid = r.get("id")
                        ridx = int(r.get("result", 0) or 0)
                        v = id_to_value.get(pid)
                        if v is not None:
                            ret_operands.append(v if ridx == 0 else v)
                    # TODO: Import function from func dialect
                    ir.Operation.create(
                        name="func.return",
                        attributes={},
                        results=[],
                        operands=ret_operands,
                    )

            # Verify the module before printing
    #       module.operation.verify()
            print(module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
