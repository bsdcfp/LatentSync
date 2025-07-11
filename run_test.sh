#!/bin/bash

#######################################
# Run Local Function Test
# Arguments:
#   -s:           打印调试信息
#   -h:           帮助信息
#######################################

usage() {
    echo "Usage: $0 [-h] [-s]"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message and exit"
    echo "  -s, Shortcut for debug message"
}

while getopts ":hs" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        s )
            PARAMS="-s"
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            usage
            exit 1
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument" 1>&2
            usage
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))

echo '>>>>>>>>>>>> Start Test <<<<<<<<<<<<<<<<<<'

python3 - $PARAMS<<-EOF
import os
import sys
from testplatform.functest.local_functest import run_local_test
testpath = os.path.dirname(os.path.abspath(__file__))+'/algo_tests'
if len(sys.argv)>1 and sys.argv[1]!='':
    run_local_test(testpath,[sys.argv[1]],framework='triton')
else:
    run_local_test(testpath,framework='triton')
EOF

echo '>>>>>>>>>>>> Finish Test <<<<<<<<<<<<<<<<<<'

