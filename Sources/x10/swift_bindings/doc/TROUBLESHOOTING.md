# Troubleshooting

To diagnose issues, we can use the execution metrics and counters provided by
X10. The first thing to check when a model is slow is to generate a metrics
report.

## Get A Metrics Report

To print a report, add a `PrintX10Metrics` call to your program:

```swift
import TensorFlow

...
PrintX10Metrics()
...
```

This will log various metrics and counters at `INFO` level.

## Understand The Metrics Report

The report includes things like:

-   How many times we trigger XLA compilations and the total time spent on
    compilation.
-   How many times we launch an XLA computation and the total time spent on
    execution.
-   How many device data handles we create / destroy etc.

This information is reported in terms of percentiles of the samples. An example
is:

```
Metric: CompileTime
  TotalSamples: 202
  Counter: 06m09s401ms746.001us
  ValueRate: 778ms572.062us / second
  Rate: 0.425201 / second
  Percentiles: 1%=001ms32.778us; 5%=001ms61.283us; 10%=001ms79.236us; 20%=001ms110.973us; 50%=001ms228.773us; 80%=001ms339.183us; 90%=001ms434.305us; 95%=002ms921.063us; 99%=21s102ms853.173us
```

We also provide counters, which are named integer variables which track internal
software status. For example:

```
Counter: CachedSyncTensors
  Value: 395
```

## Known Caveats

X10 behaves semantically like regular S4TF tensors. However, there are some
performance and completeness caveats:

1.  Degraded performance because of too many recompilations.

    XLA compilation is expensive. X10 automatically recompiles the graph every
    time new shapes are encountered, with no user intervention. Models need to
    see stabilized shapes within a few training steps and from that point no
    recompilation is needed. Additionally, the execution paths must stabilize
    quickly for the same reason: X10 recompiles when a new execution path is
    encountered. To sum up, in order to avoid recompilations:

    *   Avoid highly variable dynamic shapes. However, a low number of different
        shapes could be fine. Pad tensors to fixed sizes when possible.
    *   Avoid loops with different number of iterations between training steps.
        X10 currently unrolls loops, therefore different number of loop
        iterations translate into different (unrolled) execution paths.

2.  A small number of operations aren't supported by X10 yet.

    We currently have a handful of operations which aren't supported, either
    because there isn't a good way to express them via XLA and static shapes
    (currently just `nonZeroIndices`) or lack of known use cases (several linear
    algebra operations and multinomial initialization). While the second
    category is easy to address as needed, the first category can only be
    addressed through interoperability with the CPU, non-XLA implementation.
    Using interoperability too often has significant performance implications
    because of host round-trips and fragmenting a fully fused model in multiple
    traces. Users are therefore advised to avoid using such operations in their
    models.

    On Linux, use `XLA_SAVE_TENSORS_FILE` (documented in the next section) to
    get the Swift stack trace which called the unsupported operation. Function
    names can be manually demangled using `swift-demangle`.

## More Debugging Tools

We don't expect users to use the tools in this section to debug their models,
but they can provide additional information when filing a bug report.

### Environment Variables

There are also a number of environment variables which control the behavior of
the S4TF software stack:

*   `XLA_SAVE_TENSORS_FILE`: The path to which IR graphs will be logged during
    execution. Note that the file can become really big if the option is left
    enabled for long running programs. Remove the file before each run if you
    only want logging from the current run. Note that setting this variable has
    a substantial negative impact on performance, especially when combined with
    `XLA_LOG_GRAPH_CHANGES`.

*   `XLA_SAVE_TENSORS_FMT`: The format of the graphs stored within the
    `XLA_SAVE_TENSORS_FILE` file. Can be `text` (the default), `dot` (Graphviz
    format) or `hlo`.

*   `XLA_LOG_GRAPH_CHANGES`: If set to 1 and `XLA_SAVE_TENSORS_FILE` is set,
    log a summary of graph changes and the stack traces which created them.

*   `XLA_USE_BF16`: If set to 1, transforms all the `Float` values to BF16.
    Should only be used for debugging since we offer automatic mixed precision.

*   `XLA_USE_32BIT_LONG`: If set to 1, maps S4TF `Long` type to the XLA 32 bit
    integer type. On TPU, 64 bit integer computations are expensive, so setting
    this flag might help. Of course, the user needs to be certain that the
    values still fit in a 32 bit integer.
