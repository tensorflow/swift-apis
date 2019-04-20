# Ops and Convenience Methods

The majority of the Tensor API is implemented in terms of 'ops' that are
partitioned out to the TensorFlow graph when the compiler runs. These
ops are intentionally designed to reflect TensorFlow ops, but provide nicer
Swift syntax for accessing them. In addition to the core ops themselves,
we also define some helper function wrappers, e.g. to make things symmetric
and generally feel nice to use.

The ops themselves are defined by the primitive `#tfop(...)` syntax, here 
are some examples:
```
result = #tfop("Add", lhs, rhs)
result = #tfop("Const", dtype: Float.self, value$tensor: 4.0)
```

The first parameter to this syntax is the TensorFlow op name as a string.
After that, the inputs are specified, and then attributes are specified
with their name as the keyword argument.

Inputs and outputs must be of TensorHandle, ResourceHandle, or VariantHandle
type. These are magic types known to the compiler.
