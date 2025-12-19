// Simple test case to check basic lowering
module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @simple_test() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst = arith.constant 0.0 : f32
    %0 = wave.register %cst : vector<4xf32>
    return
  }
}