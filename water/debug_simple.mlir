// Minimal test case to debug the specific issue
module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @simple_read(
    %mem: !wave.tensor<[@M] of f16, <global>>
  ) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64}>} {
    %data = wave.read %mem index [{
      M : [#wave.index_symbol<T0>] -> (T0, 1, 1)
    }] : (!wave.tensor<[@M] of f16, <global>>) -> vector<1xf16>
    return
  }
}