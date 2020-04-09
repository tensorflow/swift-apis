VERS_1.0 {
  # Export symbols in c_api.h.
  global:
    *TF_*;
    *TFE_*;
    *swift_xla*;
    *g_trace_level*;
    *xla*primitive_util*;
    *xla*MakeSpatialPadding*;
    *xla*PaddingConfig*;
    *google*protobuf*internal*;

  # Hide everything else.
  local:
    *;
};
