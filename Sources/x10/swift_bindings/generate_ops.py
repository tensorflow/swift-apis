# Lint as: python3
from absl import app
from absl import flags
import re

import yaml

FLAGS = flags.FLAGS

flags.DEFINE_string("def_file", None, "path to list of ops")
flags.DEFINE_string("swift_out", None, "path for the generated swift file")
flags.DEFINE_string("cc_output", None, "path for the generated cc file")

HEADER = """// Autogenerated by codegen.py. Do not modify.
"""

def node_type_define(op):
   tensor_args = []
   attr_args = []
   for arg in op["args"]:
     if arg[1] == "Tensor": tensor_args.append(arg)
     else: attr_args.append(arg)
   def format_pretty_print(arg):
     return f" << \", {arg[0]}=\" << {arg[0]}_"
   def format_ctor_arg(arg):
     name, stype = arg
     if stype == "Tensor": return f"const Value& {name}"
     if stype == "Int64": return f"xla::int64 {name}"
     raise f"Problem: no such type: {stype}"
   lower_arg_i = 0
   def format_lower_arg(arg):
     nonlocal lower_arg_i
     name, stype = arg
     if stype == "Tensor":
       i = lower_arg_i
       lower_arg_i += 1
       return "loctx->GetOutputOp(operand(" + str(i) + "))"
     if stype == "Int64": return f"{name}_"
     raise f"Problem: no such type: {stype}"
   clone_arg_i = 0
   def format_clone_arg(arg):
     nonlocal clone_arg_i
     name, stype = arg
     if stype == "Tensor":
       i = clone_arg_i
       clone_arg_i += 1
       return "operands.at(" + str(i) + ")"
     if stype == "Int64": return f"{name}_"
     raise f"Problem: no such type: {stype}"
   def format_attr_define(arg):
     name, stype = arg
     if stype == "Int64": return f"  xla::int64 {name}_;\n"
     raise f"Problem: no such type: {stype}"
   def format_attr_init(arg):
     return f",\n        {arg[0]}_({arg[0]})"
   shape_fn = f"""{{}}\n#error no shape function for {op["op_node_name"]}\n"""
   def resolve_shape_fn(shape_fn):
     for arg in tensor_args:
       if arg[0] == shape_fn: return f"{arg[0]}.shape()"
     return f"""{shape_fn}({", ".join(arg[0] for arg in op["args"])})"""
   if op["shape_fn"]:
     shape_fn = resolve_shape_fn(op["shape_fn"])
   num_outputs = 1
   return f"""
class {op["op_node_name"]} : public Node {{
 public:
  {op["op_node_name"]}({", ".join(format_ctor_arg(arg) for arg in op["args"])})
      : Node(ir::OpKind({op["x10_enum"]}),
             {{{", ".join(arg[0] for arg in tensor_args)}}}, {shape_fn},
             /*num_outputs=*/{str(num_outputs)}, xla::util::MHash({", ".join(arg[0] for arg in attr_args)})){
"".join(format_attr_init(arg) for arg in attr_args)
} {{}}

  NodePtr Clone(OpList operands) const override {{
    return MakeNode<{op["op_node_name"]}>(
        {", ".join(format_clone_arg(arg) for arg in op["args"])});
  }}

  XlaOpVector Lower(LoweringContext* loctx) const override {{
    xla::XlaOp result = {op["lower_fn"]}(
        {", ".join(format_lower_arg(arg) for arg in op["args"])});
    return ReturnOp(result, loctx);
  }}

  std::string ToString() const override {{
    std::stringstream ss;
    ss << Node::ToString(){"".join(format_pretty_print(arg) for arg in attr_args)};
    return ss.str();
  }}

 private:
{"".join(format_attr_define(arg) for arg in attr_args)}}};
"""

def c_function_define(op):
  def format_arg_def(arg):
    name, stype = arg
    if stype == "Tensor": return "OpaqueXLATensor* " + name
    if stype == "Int64": return "int64_t " + name
    raise "problem unknown type: " + stype
  def format_arg_ref(arg):
    name, stype = arg
    if stype == "Tensor": return name + "_ir_value"
    for extra in op["extras"]:
      if extra[0] == "canonicalize" and extra[1] == name:
        return f"swift_xla::XlaHelpers::GetCanonicalDimensionIndex({name}, {extra[2]}_ir_value.shape().rank())"
    return name
  def unpack_arg(arg):
    name, stype = arg
    if stype == "Tensor": return f"  auto {name}_ir_value = {name}->GetIrValue();\n"
    return ""
  args = op["args"]
  first_tensor = args[0][0]
  return f"""
OpaqueXLATensor* XLATensor_{op["c_name"]}({", ".join(format_arg_def(arg) for arg in op["args"])}) {{
{"".join(unpack_arg(arg) for arg in op["args"])}  return new swift_xla::XLATensor({first_tensor}->CreateFrom(
      swift_xla::ir::MakeNode<swift_xla::ir::ops::{op["op_node_name"]}>({", ".join(format_arg_ref(arg) for arg in op["args"])})));
}}
"""

def snake_to_camel(name):
  return "".join(map(lambda x: x.capitalize(),name.split("_")))

def canonicalize_op(op):
  tokens = re.findall("(\w+|[\(\),:]|->)", op["def"])
  op["c_name"] = tokens[0]
  def expect(cond):
    if not cond: raise ValueError(f"""invalid format: {repr(op["def"])}""")
  expect(tokens[1] == '(')
  def isWord(idx):
    return re.match("\w+", tokens[idx]) != None
  i = 2
  args = []
  if tokens[i] != ')':
    while True:
      expect(tokens[i + 1] == ':')
      expect(isWord(i) and isWord(i + 2))
      args.append((tokens[i], tokens[i + 2]))
      i += 3
      if tokens[i] == ')': break
      expect(tokens[i] == ',')
      i += 1
  i += 1

  op["args"] = args
  if "op_node_name" not in op: op["op_node_name"] = snake_to_camel(op["c_name"])
  op["extras"] = [a.split() for a in op["extras"]]
  del op["def"]

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  op_list = yaml.full_load(open(FLAGS.def_file).read())
  for op in op_list: canonicalize_op(op)

  open(FLAGS.cc_output, "w+").write(HEADER + """
namespace swift_xla {
namespace ir {
namespace ops {
namespace {
""" + ("".join(node_type_define(op) for op in op_list)) + """
}  // namespace
}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
""" + "".join(c_function_define(op) for op in op_list))

if __name__ == "__main__":
  app.run(main)
