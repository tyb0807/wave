// Test case for wave.iterate with wave.read and wave.write operations
// This should now work with our single-phase conversion fix

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @test_iterate_with_read_write(
    %mem_in: !wave.tensor<[@M, @K] of f16, <global>>,
    %mem_out: !wave.tensor<[@M, @K] of f16, <global>>
  ) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_K = 32, K = 128, M = 64}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %result = wave.iterate @K iter_args(%mem_out) {
    ^bb0(%arg0: !wave.tensor<[@M, @K] of f16, <global>>):
      // Read from input memory
      %data = wave.read %mem_in index [{
        M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 * 64 + T0, 1, 1),
        K : [#wave.index_symbol<WG1>, #wave.index_symbol<T1>] -> (WG1 * 32 + T1, 1, 1)
      }] : (!wave.tensor<[@M, @K] of f16, <global>>) -> vector<1xf16>

      // Write to output memory
      wave.write %data, %arg0 index [{
        M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 * 64 + T0, 1, 1),
        K : [#wave.index_symbol<WG1>, #wave.index_symbol<T1>] -> (WG1 * 32 + T1, 1, 1)
      }] : vector<1xf16>, !wave.tensor<[@M, @K] of f16, <global>>

      wave.yield %arg0 : !wave.tensor<[@M, @K] of f16, <global>>
    } : (!wave.tensor<[@M, @K] of f16, <global>>) -> !wave.tensor<[@M, @K] of f16, <global>>

    return
  }
}