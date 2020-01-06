"""
Runs "swift build" in a variety of configurations, and outputs timing
information to an xUnit file.
"""


import argparse
import psutil
import subprocess
import tempfile
import time


from junit_xml import TestCase, TestSuite


def kill(pid):
  proc = psutil.Process(pid)
  for child in proc.children(recursive=True):
    child.kill()
  proc.kill()


def execute_benchmark(test_case, cmd, timeout):
  start = time.time()
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  try:
    stdout, stderr = proc.communicate(timeout=timeout)
    code = proc.returncode
  except subprocess.TimeoutExpired as e:
    kill(proc.pid)
    stdout, stderr = proc.communicate()
    code = 0
    test_case.add_failure_info(str(e))

  end = time.time()
  test_case.stdout = stdout
  test_case.stderr = stderr
  test_case.elapsed_sec = end - start

  if code != 0:
    test_case.add_failure_info('Nonzero exit code: %d' % code)
    return test_case

  return test_case


def benchmark(test_case, cmd, timeout=600):
  with tempfile.TemporaryDirectory() as build_path:
    cmd += ['--build-path', build_path]
    return execute_benchmark(test_case, cmd, timeout)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('swift', help='path to swift executable')
  parser.add_argument('output', help='where to write xUnit output')
  args = parser.parse_args()

  test_cases = [
      benchmark(
          TestCase('debug build'),
          [args.swift, 'build', '--product', 'TensorFlow']
      ),
      benchmark(
          TestCase('release build'),
          [args.swift, 'build', '-c', 'release', '--product', 'TensorFlow']
      ),

      # The point of "release build -Onone" is to compile TensorFlow in
      # "-whole-module-optimization" mode without "-O".
      benchmark(
          TestCase('release build -Onone'),
          [args.swift, 'build', '-c', 'release', '--product', 'TensorFlow',
           '-Xswiftc', '-Onone']
      ),
  ]

  test_suite = TestSuite('swift-apis compile time', test_cases)

  with open(args.output, 'w') as f:
    TestSuite.to_file(f, [test_suite])


if __name__ == '__main__':
  main()
