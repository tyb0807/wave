// Simple read/write case to test basic lowering
module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @simple_read_write(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{M = 32, N = 32}>
  } {
    // Simple read operation
    %data = wave.read %mem index [{
      M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 * 32 + T0, 1, 1),
      N : [#wave.index_symbol<T1>] -> (T1, 1, 1)
    }] : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<1xf16>

    // Simple write operation
    wave.write %data, %mem index [{
      M : [#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 * 32 + T0, 1, 1),
      N : [#wave.index_symbol<T1>] -> (T1, 1, 1)
    }] : vector<1xf16>, !wave.tensor<[@M, @N] of f16, <global>>

    return
  }
}