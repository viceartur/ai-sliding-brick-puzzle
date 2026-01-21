#!/bin/sh

COMMAND=$1
FILE=$2
EXTRA=$3

case "$COMMAND" in
    print)
        python3 sbp.py print "$FILE"
        ;;
    done)
        python3 sbp.py done "$FILE"
        ;;
    availableMoves)
        python3 sbp.py availableMoves "$FILE"
        ;;
    applyMove)
        python3 sbp.py applyMove "$FILE" "$EXTRA"
        ;;
    compare)
        python3 sbp.py compare "$FILE" "$EXTRA"
        ;;
    norm)
        python3 sbp.py norm "$FILE"
        ;;
    random)
        python3 sbp.py random "$FILE" "$EXTRA"
        ;;
esac