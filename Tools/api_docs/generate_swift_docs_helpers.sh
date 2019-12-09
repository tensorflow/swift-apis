#!/usr/bin/env bash
#
# Generates Swift reference docs using jazzy.

################################################################################
# Shell Options
################################################################################
set -o errexit
set -o nounset

################################################################################
# Global Variables
################################################################################
# The colors for displaying text in the terminal
if [[ "${TERM}" = *"xterm"* || "${TERM}" = *"screen"* ]]; then
  RESET_ATTR=$(tput sgr0)
  RED=$(tput setaf 1)
  GREEN=$(tput setaf 2)
fi

################################################################################
# Displays an error message and exits if a command fails.
#
# Preconditions:
#   None
# Postconditions:
#   None
# Globals:
#   RED
#   RESET_ATTR
# Parameters:
#   @param error_message  The error message displayed in the terminal.
# Returns:
#   None
################################################################################
print_error_message() {
  printf "${RED-}ERROR: ${RESET_ATTR-}"
  echo "$1"
  echo "Exiting..."
  exit 1
}

################################################################################
# Checks if jazzy prerequisites are installed locally. If a prerequisite is
# missing, an error message is displayed with instructions for installing
# the missing prerequisite. If jazzy is installed locally, the function
# checks if the latest version of jazzy is installed. If it's not, the user
# is prompted to update jazzy (updating to the latest version of jazzy is
# required).
#
# Prerequisites for using jazzy:
#   1) RubyGems
#   2) jazzy
#   3) Xcode Command Line Tools
#   4) Xcode
#
# Preconditions:
#   None
# Postconditions:
#   None
# Globals:
#   GREEN
#   RESET_ATTR
# Parameters:
#   None
# Returns:
#   None
################################################################################
check_jazzy_prerequisites() {
  # Checks if RubyGems is installed locally
  gem --version
  if [[ "$?" -ne 0 ]]; then
    print_error_message "In order to generate Swift and Objective-C reference docs,
      you need RubyGems installed locally. Visit https://rubygems.org/pages/download for
      instructions on how to install RubyGems."
  fi

  # Checks if jazzy is installed locally
  jazzy --version
  if [[ "$?" -ne 0 ]]; then
    print_error_message "In order to generate Swift and Objective-C reference docs,
      you need jazzy installed locally. To install jazzy, run: gem install jazzy"
  fi

  # Checks if Xcode Command Line Tools is installed locally
  xcrun --version
  if [[ "$?" -ne 0 ]]; then
    print_error_message "In order to generate Swift and Objective-C reference docs,
      you need Xcode Command Line Tools installed locally. To install Xcode Command Line Tools,
      run the following command in Terminal: [sudo] xcode-select --install
      You will also need to install the latest version of Xcode."
  fi
}

################################################################################
# Generates Swift reference docs in a single docset using jazzy.
# This should be used for Swift-only projects.
#
# Preconditions:
#   None
# Postconditions:
#   None
# Parameters:
#   @param output_folder  The folder name to output the HTML docs to.
#   @param templates_dirpath  The google3 directory path to the custom mustache
#                             templates.
#   @param assets_dirpath  The google3 directory path to custom site assets.
#   @param config_file  The jazzy configuration file.
#   @param preprocess_script  The documentation preprocessing script.
#   @param module_name  The name of the module (i.e. framework) being
#                       documented.
#   @param custom_head_html  The custom HTML string to inject in <head></head>.
#   @param docset_path  The relative DevSite path to where the generated docs
#                       will be located.
# Returns:
#   None
################################################################################
jazzy_generate_swift_docs() {
  # Verify the correct number of arguments were passed to this function.
  local args
  args=9
  if [[ "$#" -ne "${args}" ]]; then
    print_error_message "jazzy_generate_docs() takes ${args} arguments - please
    make sure to pass the correct number of arguments to jazzy_generate_docs()."
  fi

  # Checks if all Jazzy prerequisites are satisfied.
  check_jazzy_prerequisites

  # Create the Jazzy directory that will contain temporary artifacts.
  if [[ -d "Jazzy" ]]; then
    # NOTE: The `go/jazzy` version of this script prints an error messagee if
    # the "Jazzy" temporary directory already exists.
    # This version of the script simply deletes the temporary directory, since
    # it is generated and unlikely to have useful contents.
    echo "Deleting old Jazzy directory at $(pwd)."
    rm -rf Jazzy
    # print_error_message "A Jazzy directory already exists in $(pwd). Please
    # remove this directory before calling jazzy_generate_docs()."
  fi
  mkdir Jazzy
  pushd Jazzy

  # Delete old API docs files. Persist files like index.md/README.md.
  persisted_files=(index.md README.md)
  for f in "${persisted_files[@]}"; do
    if [[ -f "../$1/$f" ]]; then
      cp "../$1/$f" "$PWD"
    fi
  done
  # Delete output directory.
  # rm -rf "../$1"

  directory="$PWD"
  local_dir="$PWD/tmp/jazzy/$6"
  rm -rf "${local_dir}"
  mkdir -p "${local_dir}"

  # Copy the mustache templates to the Jazzy directory.
  cp -r "../$2"/ ./templates
  # Copy assets to the Jazzy directory.
  cp -r "../$3"/ ./assets
  # Copy the configuration file to the Jazzy directory.
  cp -r "../$4" "${local_dir}/.jazzy.yaml"
  pwd

  pushd "${local_dir}"
  # Create Swift library package and delete dummy source files.
  swift package init
  swift package tools-version --set-current
  rm -rf "Sources/$6"/*
  rm -rf "Tests/$6"/*

  # Get swift and swift-apis directories.
  swift_build_dir="$8"
  swift_dir="${swift_build_dir}/swift"
  swift_apis_dir="${swift_build_dir}/tensorflow-swift-apis"

  # TODO(b/126777279): Handle duplicate filenames across modules.

  # Copy sources from Swift.
  # Python module.
  cp "${swift_dir}/stdlib/public/Python/"*.swift "Sources/$6"
  # Automatic differentiation APIs.
  cp "${swift_dir}/stdlib/public/Differentiation/"*.swift "Sources/$6"
  cp "${swift_dir}/stdlib/public/core/KeyPathIterable.swift" "Sources/$6"

  # Copy sources from swift-apis.
  cp "${swift_apis_dir}/Sources"/**/*.swift "Sources/$6"

  # Generate the .gyb files.
  find . -name "*.gyb" |
  while read f; do
      trimmed=${f%.*}
      "${swift_dir}/utils/gyb" -o "${trimmed}" "${f}"
  done

  # Clean Swift source files.
  # NOTE(b/126775893): All double-slash comments are removed from files to
  # prevent lost documentation comments.
  find . -name "*.swift" |
  while read f; do
      sed -E -i "" "
          /@_inlineable[ ]*\n?/s///
          /@_versioned[ ]*\n?/s///
          /@inlinable[ ]*\n?/s///
          /@usableFromInline[ ]*\n?/s///
          /@inline\(__always\)[ ]*\n?/s///
          /@inline\(__never\)[ ]*\n?/s///
          /^[[:blank:]]*\/\/[^\/]/d
          " "${f}"
  done

  # Generate xcodeproj.
  swift package tools-version --set 5.1
  swift package generate-xcodeproj

  # Generate Swift reference docs using Jazzy.
  # NOTE(b/126780157): `jazzy` command is flaky. Repeat it until success.
  until jazzy \
    --output "${directory}/$1" \
    --theme "${directory}" \
    --config .jazzy.yaml \
    --head "$7" \
    --author "$9/$1" \
    --undocumented-text ""
  do
    echo "Jazzy failed; trying again."
  done
  popd

  if [[ "$?" -ne 0 ]]; then
    rm -rf Jazzy
    print_error_message "jazzy failed to generate Swift reference docs."
  fi
  # Post-process the Jazzy generated reference docs.
  mkdir -p "$1"
  post_process_jazzy_docs "$1"
  # Copy the css directory to the generated docs directory.
  cp -r assets/css "$1"/css
  popd
  pwd
  cp -r Jazzy/"$1"/. "$1"/
  rm -rf Jazzy
}

################################################################################
# Performs post-processing of the jazzy generated docs. Removes jazzy generated
# directories and files that are not necessary. Replaces spaces with
# "-" in directory and file names.
#
# Preconditions:
#   None
# Postconditions:
#   None
# Globals:
#   None
# Parameters:
#   @param output_folder  The output folder that contains the jazzy
#                         generated docs.
# Returns:
#   None
################################################################################
post_process_jazzy_docs() {
  # Replace spaces with "-" in directory and file names.
  for f in "$1"/*\ *; do
    if [[ -d "${f}" || -f "${f}" ]]; then
      mv "${f}" "${f// /-}"
    fi
  done

  # TODO: Handle leading underscore files.
  # Currently, they are processed by Jazzy and show up in _toc.yaml, but are
  # hidden by devsite, leading to 404 errors.

  # Remove generated directories and files
  rm -rf "$1"/docsets
  rm -rf "$1"/js
  rm -rf "$1"/css
  rm -rf "$1"/img
  rm -f "$1"/undocumented.json
  rm -f "$1"/search.json
  # Move the nav.mustache output to a _toc.yaml file
  touch "$1"/_toc.yaml
  sed -e "1,/<\/html>/d" "$1"/index.html > "$1"/_toc.yaml
  # Replace URL encoded space character with "-".
  sed -i "" "s/%20/-/g" "$1"/_toc.yaml
  rm -f "$1"/index.html
  # Clean up HTML files.
  find . -name "*.html" |
  while read f; do
    sed -E -i "" "
      /<pre class=\"highlight ([a-z]+)?\">/s//<pre class=\"prettyprint\">/
      /toc:/,\$d
      /header-anchor/d
      /<p>.*$1<\/p>/d
      /(.*Type) (Definitions.*)/s//\1-\2/
      " "${f}"
  done
  # Clean up YAML files.
  find . -name "*.yaml" |
  while read f; do
    sed -i "" "
      /\(: .*\) \(.*\)/s//\1-\2/
      /\(title: .*\)-\(.*\)/s//\1 \2/
      " "${f}"
  done
}
