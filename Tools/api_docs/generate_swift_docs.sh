#!/usr/bin/env bash
#
# API doc generation script for Swift for TensorFlow.
# This script is adapted from scripts described at go/jazzy.
#
# Visit go/jazzy to learn more about using Jazzy.

USAGE="usage: generate_swift_docs.sh <swift root directory>"
if [[ $# -eq 0 ]]; then
  echo "$USAGE"
  exit 1
elif [[ ! -d "$1" ]]; then
  echo "'$1' does not exist."
  exit 1
fi
SWIFT_SOURCES="$1"

## Import Jazzy helper functions.
. generate_swift_docs_helpers.sh

## Set global variables.
JAZZY_OUTPUT_FOLDER="api_docs"
JAZZY_TEMPLATES="templates"
JAZZY_ASSETS="assets"
JAZZY_CONFIG=".jazzy.yaml"
JAZZY_PREPROCESS_SCRIPT="devsite/tools/tensorflow/subsites/tools/swift/preprocess_docs.py"
JAZZY_MODULE_NAME="TensorFlow"
DOCSET_PATH="swift"
JAZZY_CUSTOM_HEAD_HTML="<meta name=\"project_path\" value=\"/swift/_project.yaml\" />
    <meta name=\"book_path\" value=\"/swift/_book.yaml\" />"
TF_DOCSET="third_party/devsite/tensorflow/en/swift"

# Enable debug mode.
set -x

## Generate Swift reference docs.
# Arguments:
#   $1 - Folder name to output the HTML docs to
#   $2 - Path to the custom mustache templates
#   $3 - Path to the custom assets folder (CSS)
#   $4 - Path to the Jazzy configuration file (.jazzy.yaml)
#   $5 - Path to the doc preprocessing script.
#   $6 - Name of the module (i.e. framework) being documented
#   $7 - Custom HTML string to inject in <head></head>
#   $8 - Root directory of the Swift repository.
#   $9 - Relative DevSite path to where the generated docs will be located
jazzy_generate_swift_docs \
  "${JAZZY_OUTPUT_FOLDER}" \
  "${JAZZY_TEMPLATES}" \
  "${JAZZY_ASSETS}" \
  "${JAZZY_CONFIG}" \
  "${JAZZY_PREPROCESS_SCRIPT}" \
  "${JAZZY_MODULE_NAME}" \
  "${JAZZY_CUSTOM_HEAD_HTML}" \
  "${SWIFT_SOURCES}" \
  "${DOCSET_PATH}"
