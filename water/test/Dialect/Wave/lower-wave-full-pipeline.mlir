// RUN: water-opt %s -allow-unregistered-dialect --pass-pipeline="builtin.module(lower-wave-control-flow,lower-wave-to-mlir)" --split-input-file | FileCheck %s

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @test_simple_iteration() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, M = 64, BLOCK_K = 16}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = memref.alloc() : memref<64xf32, #gpu.address_space<workgroup>>
    %0 = builtin.unrealized_conversion_cast %alloc : memref<64xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@M] of f32, <shared>>

    // CHECK-LABEL: func.func @test_simple_iteration
    // CHECK-NOT: wave.iterate
    // CHECK-NOT: iterator = #wave.symbol
    // CHECK: %[[LB:.*]] = arith.constant 0 : index
    // CHECK: %[[UB:.*]] = arith.constant 4 : index
    // CHECK: %[[STEP:.*]] = arith.constant 1 : index
    // CHECK: %{{.*}} = scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (memref<64xf32, #gpu.address_space<workgroup>>) {
    // CHECK: scf.yield %{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
    // CHECK: }
    %result = wave.iterate @K iter_args(%0) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> (!wave.tensor<[@M] of f32, <shared>>)

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @test_iteration_with_operations() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, M = 32, BLOCK_K = 16}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
    %0 = builtin.unrealized_conversion_cast %alloc : memref<32xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@M] of f32, <shared>>

    // CHECK-LABEL: func.func @test_iteration_with_operations
    // CHECK-NOT: wave.iterate
    // CHECK-NOT: wave.register
    // CHECK-NOT: iterator = #wave.symbol
    // CHECK: %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (memref<32xf32, #gpu.address_space<workgroup>>) {
    // CHECK: arith.constant 1.000000e+00 : f32
    // CHECK: scf.yield %{{.*}} : memref<32xf32, #gpu.address_space<workgroup>>
    // CHECK: }
    %result = wave.iterate @K iter_args(%0) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      %cst = arith.constant 1.0 : f32
      %reg = wave.register %cst : vector<4xf32>
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> (!wave.tensor<[@M] of f32, <shared>>)

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @test_iteration_with_read_write(%mem: memref<64x64xf32, #gpu.address_space<global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, M = 64, BLOCK_K = 16}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = memref.alloc() : memref<64xf32, #gpu.address_space<workgroup>>
    %0 = builtin.unrealized_conversion_cast %alloc : memref<64xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@M] of f32, <shared>>
    %1 = builtin.unrealized_conversion_cast %mem : memref<64x64xf32, #gpu.address_space<global>> to !wave.tensor<[@M, @K] of f32, <global>>

    // CHECK-LABEL: func.func @test_iteration_with_read_write
    // CHECK-NOT: wave.iterate
    // CHECK-NOT: wave.register
    // CHECK-NOT: iterator = #wave.symbol
    // CHECK: %[[LB:.*]] = arith.constant 0 : index
    // CHECK: %[[UB:.*]] = arith.constant 4 : index
    // CHECK: %[[STEP:.*]] = arith.constant 1 : index
    // CHECK: %{{.*}} = scf.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (memref<64xf32, #gpu.address_space<workgroup>>) {
    // CHECK: arith.constant 2.000000e+00 : f32
    // CHECK: scf.yield %{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
    // CHECK: }
    %result = wave.iterate @K iter_args(%0) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      %cst = arith.constant 2.0 : f32
      %reg = wave.register %cst : vector<4xf32>
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> (!wave.tensor<[@M] of f32, <shared>>)

    return
  }
}