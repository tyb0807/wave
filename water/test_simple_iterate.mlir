// Simple iterate case without read/write to test region movement
module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @simple_iterate() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 128, BLOCK_K = 32}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %cst = arith.constant 1.0 : f32
    %init_reg = wave.register %cst : vector<4xf32>
    %result = wave.iterate @K iter_args(%init_reg) {
    ^bb0(%arg0: vector<4xf32>):
      %cst2 = arith.constant 2.0 : f32
      %reg = wave.register %cst2 : vector<4xf32>
      wave.yield %reg : vector<4xf32>
    } : (vector<4xf32>) -> vector<4xf32>
    return
  }
}