# REQUIRES: water
# RUN: python %s | FileCheck %s

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from wave_lang.kernel._support.indexing import IndexSymbol
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.mlir_converter.mlir_converter import emit_wave_dialect
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.water import get_water_opt
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)


@run_test
def mlir_converter_simple_with_lowering():
    """Test MLIR converter with simple kernel and lower-wave-to-mlir pass."""
    import torch
    from torch.testing import assert_close
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    # Simple matrix addition kernel to reduce complexity
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE_A = tkl.sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = tkl.sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = tkl.sym.ADDRESS_SPACE_C

    # Use smaller sizes for testing
    shape = (32, 32)
    wave_size = 64

    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, shape[0], 0),
        tkw.WorkgroupConstraint(N, shape[1], 1),
        tkw.WaveConstraint(M, shape[0]),
        tkw.WaveConstraint(N, shape[1]),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: shape[0], N: shape[1]},
        ),
    ]

    @wave.wave(constraints)
    def matrix_add(
        a: Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
        b: Memory[M, N, ADDRESS_SPACE_B, tkl.f16],
        c: Memory[M, N, ADDRESS_SPACE_C, tkl.f16],
    ):
        # loads values from memory into registers
        a_reg = wave.read(a)
        b_reg = wave.read(b)

        # compute the sum
        c_reg = a_reg + b_reg

        # writing results back to memory
        wave.write(c_reg, c)

    # Set parameters for compilation
    subs: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        M: shape[0],
        N: shape[1],
    }

    # First test: MLIR generation only (existing functionality)
    options_mlir = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,  # Avoid IREE compilation
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        print_mlir=True,
    )
    options_mlir = set_default_run_config(options_mlir)

    compiled_kernel = wave_compile(options_mlir, matrix_add)

    # Get the trace from the compiled kernel
    trace = compiled_kernel.compiled_graph
    constraints = matrix_add.constraints

    # Use the mlir_converter to emit wave MLIR dialect
    wave_dialect_mlir, diagnostics = emit_wave_dialect(
        trace, constraints, options, False
    )

    if diagnostics:
        print(diagnostics)
    assert (
        len(diagnostics) == 0
    ), "dialect emission should create valid IR, therefore diagnostics should be empty"

    # Print the Wave dialect MLIR first
    print("=== Wave Dialect MLIR (before normal form) ===")
    print(wave_dialect_mlir)

    # Add required normal form attribute for lower-wave-to-mlir pass
    # TODO: This should be done by a proper pass, but for now use string manipulation
    normal_form_attr = "wave.normal_form = #wave.normal_form<full_types>"

    if "module {" in wave_dialect_mlir:
        # Simple module with no existing attributes
        wave_dialect_mlir = wave_dialect_mlir.replace(
            "module {",
            f"module attributes {{{normal_form_attr}}} {{"
        )
    elif "module attributes" in wave_dialect_mlir:
        # Module already has attributes - insert the normal form attribute
        # Pattern: module attributes {existing_attrs} {
        import re
        def add_normal_form(match):
            existing_attrs = match.group(1).strip()
            if existing_attrs and not existing_attrs.endswith(','):
                return f"module attributes {{{normal_form_attr}, {existing_attrs}}} {{"
            else:
                return f"module attributes {{{normal_form_attr}{existing_attrs}}} {{"

        wave_dialect_mlir = re.sub(
            r"module attributes \{([^}]*)\} \{",
            add_normal_form,
            wave_dialect_mlir
        )

    print("\n=== Wave Dialect MLIR (with normal form) ===")
    print(wave_dialect_mlir)

    # Apply passes in two steps to see intermediate results
    print("\n=== Applying two-pass lowering pipeline ===")

    water_opt_path = get_water_opt()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(wave_dialect_mlir)
        mlir_file_path = f.name

    try:
        # Step 1: Run propagate-elements-per-thread pass
        print("\n=== Step 1: Running propagate-elements-per-thread pass ===")
        result1 = subprocess.run([
            water_opt_path,
            mlir_file_path,
            "--allow-unregistered-dialect",
            "--water-wave-propagate-elements-per-thread"
        ], capture_output=True, text=True)

        if result1.returncode != 0:
            print(f"Step 1 failed with return code {result1.returncode}")
            print(f"stderr: {result1.stderr}")
            print(f"stdout: {result1.stdout}")
            raise RuntimeError(f"Step 1 failed: {result1.stderr}")

        intermediate_mlir = result1.stdout
        print(intermediate_mlir)

        # Write intermediate result to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(intermediate_mlir)
            intermediate_file_path = f.name

        # Step 2: Run lower-wave-to-mlir pass on intermediate result
        print("\n=== Step 2: Running lower-wave-to-mlir pass ===")
        result2 = subprocess.run([
            water_opt_path,
            intermediate_file_path,
            "--allow-unregistered-dialect",
            "--lower-wave-to-mlir",
            "--canonicalize",
            "--cse",
            "--mlir-print-local-scope"
        ], capture_output=True, text=True)

        if result2.returncode != 0:
            print(f"Step 2 failed with return code {result2.returncode}")
            print(f"stderr: {result2.stderr}")
            print(f"stdout: {result2.stdout}")
            raise RuntimeError(f"Step 2 failed: {result2.stderr}")

        lowered_mlir = result2.stdout
        print(lowered_mlir)

        # Clean Wave-specific attributes that cause issues in water-opt
        def clean_wave_attributes(mlir_text: str) -> str:
            import re
            # Remove wave.normal_form attribute from module attributes
            mlir_text = re.sub(r',\s*wave\.normal_form\s*=\s*#wave\.normal_form<[^>]*>', '', mlir_text)
            mlir_text = re.sub(r'wave\.normal_form\s*=\s*#wave\.normal_form<[^>]*>,?\s*', '', mlir_text)
            return mlir_text

        lowered_mlir = clean_wave_attributes(lowered_mlir)
        print("Cleaned lowered MLIR:")
        print(lowered_mlir)

        # Clean up intermediate file
        Path(intermediate_file_path).unlink()

    finally:
        # Clean up temporary file
        Path(mlir_file_path).unlink()

    # Second test: End-to-end execution with water pipeline
    print("\n=== End-to-End Execution Test ===")

    # Create test tensors
    a_tensor = device_randn(shape, dtype=torch.float16)
    b_tensor = device_randn(shape, dtype=torch.float16)
    c_tensor = device_zeros(shape, dtype=torch.float16)

    # Expected result (CPU computation)
    expected = a_tensor + b_tensor

    print(f"Input tensors: a.shape={a_tensor.shape}, b.shape={b_tensor.shape}")
    print(f"Expected sum shape: {expected.shape}")

    # Test actual water pipeline execution with override MLIR
    options_e2e = WaveCompileOptions(
        subs=subs,
        use_water_pipeline=True,  # Use water pipeline for compilation and execution
        canonicalize=True,
        override_mlir=lowered_mlir,  # Use the Water-generated MLIR
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options_e2e = set_default_run_config(options_e2e)

    # Always use a valid GPU target for water pipeline testing
    gpu_available = torch.cuda.is_available()
    if options_e2e.target == "cpu" or not gpu_available:
        original_target = options_e2e.target
        options_e2e.target = "gfx90a"  # Use a common AMD GPU target for testing
        print(f"GPU available: {gpu_available}")
        print(f"Target device: {options_e2e.device}")
        print(f"Target architecture: {original_target} -> {options_e2e.target} (overridden for water pipeline)")
    else:
        print(f"GPU available: {gpu_available}")
        print(f"Target device: {options_e2e.device}")
        print(f"Target architecture: {options_e2e.target}")

    print("Compiling kernel with water pipeline using override MLIR...")
    try:
        compiled_e2e = wave_compile(options_e2e, matrix_add)
        print("✓ Water pipeline compilation PASSED!")

        if gpu_available:
            print("Executing kernel with water pipeline...")
            compiled_e2e(a_tensor, b_tensor, c_tensor)
            print("✓ Water pipeline execution PASSED!")

            print("Verifying computational results...")
            print(f"Result shape: {c_tensor.shape}")
            print(f"Sample values - Expected[0,0] = {expected[0,0]:.4f}, Got[0,0] = {c_tensor[0,0]:.4f}")
            print(f"Sample values - Expected[1,1] = {expected[1,1]:.4f}, Got[1,1] = {c_tensor[1,1]:.4f}")

            # Verify correctness with tolerance for FP16
            assert_close(c_tensor, expected, rtol=1e-2, atol=1e-2)
            print("✓ End-to-end water pipeline test PASSED - results are mathematically correct!")

            # Additional verification
            max_diff = torch.max(torch.abs(c_tensor - expected)).item()
            mean_diff = torch.mean(torch.abs(c_tensor - expected)).item()
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"Mean absolute difference: {mean_diff:.6f}")
        else:
            print("GPU not available - skipping execution, but compilation test PASSED!")
            print("✓ Water pipeline compilation with valid GPU target PASSED!")

    except Exception as e:
        print(f"✗ Water pipeline execution FAILED: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()

        # If it's a library issue, provide helpful info
        if "library not found" in str(e):
            print("\nNote: This may be due to missing Wave runtime libraries.")
            print("Make sure Wave is properly built with: cmake --build build --target all")

        raise

    # CHECK-LABEL: mlir_converter_simple_with_lowering
    # CHECK: === Wave Dialect MLIR (before normal form) ===
    # CHECK: module {
    # CHECK: func.func @kernel(
    # CHECK-SAME: !wave.tensor<[@M, @N] of f16, <global>>

    # CHECK: === Wave Dialect MLIR (with normal form) ===
    # CHECK: module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>}
    # CHECK: func.func @kernel(
    # Wave dialect operations should be present:
    # CHECK: wave.read
    # CHECK: wave.add
    # CHECK: wave.write

    # CHECK: === Applying two-pass lowering pipeline ===
    # CHECK: === Step 1: Running propagate-elements-per-thread pass ===
    # After step 1, register tensors should be converted to vectors:
    # CHECK: vector<16xf16>
    # CHECK: wave.normal_form = #wave.normal_form<memory_only_types>
    # CHECK: === Step 2: Running lower-wave-to-mlir pass ===
    # After step 2, Wave dialect operations should be converted to standard MLIR:
    # CHECK: func.func
    # CHECK: arith.constant
    # CHECK-NOT: wave.add
    # CHECK: arith.addf

    # CHECK: === End-to-End Execution Test ===
    # CHECK: Input tensors: a.shape=torch.Size([32, 32]), b.shape=torch.Size([32, 32])
    # CHECK: Expected sum shape: torch.Size([32, 32])
    # CHECK: Compiling kernel with water pipeline...
    # CHECK: Executing kernel...
    # CHECK: Verifying results...
    # CHECK: ✓ End-to-end test PASSED - results match expected values!
