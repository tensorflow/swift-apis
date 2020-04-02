import os
import re
import shutil
import sys


def _silenced(op):
    def inner(*args, **kwargs):
        try:
            return op(*args, **kwargs)
        except OSError:
            pass
    return inner


# Remove the first components of a path.
def _remove_first_path_components(root, prefix_len):
    return os.path.sep.join(root.split(os.path.sep)[prefix_len:])


# Translate given path to destination.
def _translate_path(root, dest, prefix_len):
    return os.path.join(dest, _remove_first_path_components(root, prefix_len))


def _copy_x10_headers(source, dest, name_pattern):
    _silenced(os.makedirs)(dest)
    prefix_len = len(os.path.normpath(source).split(os.path.sep))
    for root, dirs, files in os.walk(source, topdown=True, followlinks=True):
        for d in dirs:
            dest_subdir = _translate_path(root, dest, prefix_len)
            _silenced(os.mkdir)(os.path.join(dest_subdir, d))
    for root, dirs, files in os.walk(source, topdown=True, followlinks=True):
        for f in files:
            if name_pattern is None or re.match(name_pattern, f):
                dest_subdir = _translate_path(root, dest, prefix_len)
                shutil.copy(os.path.join(root, f), dest_subdir)


def _collect_headers(source_dir, x10_inc):
    _silenced(shutil.rmtree)(x10_inc)
    _copy_x10_headers(os.path.join(source_dir, 'bazel-tensorflow'),
                      x10_inc, r'.*\.h$')
    _copy_x10_headers(os.path.join(source_dir, 'bazel-bin'), x10_inc,
                      r'.*\.h$')
    _copy_x10_headers(os.path.join(source_dir, 'bazel-tensorflow',
                                   'external', 'com_google_protobuf',
                                   'src'),
                      x10_inc, r'(.*\.inc$|.*\.h)$')
    _copy_x10_headers(os.path.join(source_dir, 'bazel-tensorflow',
                                   'external', 'com_google_absl'),
                      x10_inc, r'(.*\.inc$|.*\.h)$')
    _copy_x10_headers(os.path.join(source_dir, 'bazel-tensorflow',
                                   'external', 'eigen_archive', 'Eigen'),
                      os.path.join(x10_inc, 'Eigen'), None)
    _copy_x10_headers(os.path.join(source_dir, 'third_party',
                                   'eigen3'),
                      os.path.join(x10_inc, 'third_party', 'eigen3'), None)
    _copy_x10_headers(os.path.join(source_dir, 'bazel-tensorflow',
                                   'external', 'eigen_archive',
                                   'unsupported'),
                      os.path.join(x10_inc, 'unsupported'), None)
    _silenced(shutil.rmtree)(os.path.join(x10_inc, 'external'))
    _silenced(shutil.rmtree)(os.path.join(x10_inc, 'bazel-out'))


def main(source_dir, x10_inc):
    _collect_headers(source_dir, x10_inc)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
