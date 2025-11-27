# Task Completion Checklist

## Before Completing a Task
1. **Code Quality Checks**
   - Run `cargo fmt` to ensure code formatting is consistent
   - Run `cargo clippy` to identify and fix any code improvements or warnings
   - Run `cargo check` to ensure code compiles without errors

2. **Testing**
   - Run `cargo test` to ensure all tests pass
   - If adding new functionality, ensure adequate test coverage exists
   - Test examples to ensure they still work properly

3. **Documentation**
   - Ensure all new public functions, traits, and structs have documentation
   - Update examples if needed to reflect changes
   - Follow the existing documentation style in the project

4. **Adherence to Project Patterns**
   - Follow the state update order defined in `state_update_order.md` if working with optimizers
   - Maintain consistency with existing code architecture and patterns
   - Ensure new code follows the `OptModel` and `LocalSearchOptimizer` trait contracts

## After Making Changes
1. **Verification**
   - Verify that the changes work as expected against the examples
   - Ensure the library still builds correctly for both regular and WASM targets
   - Check that parallelization (using Rayon) is handled properly if relevant

2. **Performance Considerations**
   - Ensure any changes don't negatively impact performance
   - Verify that parallelization still works as expected
   - Check for any unnecessary cloning or performance bottlenecks

3. **Final Validation**
   - Run the complete test suite one more time
   - Verify examples still run correctly
   - Ensure documentation builds without warnings