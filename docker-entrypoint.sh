#!/bin/sh
set -eu

APP_ROOT="${APP_ROOT:-/app}"

usage() {
    cat <<'EOF'
Usage:
  docker run [docker-options] IMAGE STRATEGY [OPERATION] [operation-options]
  docker run [docker-options] IMAGE setup-data STRATEGY [setup-options]
  docker run [docker-options] IMAGE paper-both [--date YYYY-MM-DD]

Strategies and operations:
  mispricing [backtest]       Run the settlement-value holdout backtest
  mispricing train            Train, calibrate, and tune the model
  mispricing prepare          Rebuild prepared data from normalized inputs
  mispricing pipeline         Prepare, train, then backtest
  mispricing paper            Score a JSONL decision stream
  mispricing live-paper       Continuously run today's and future slates
  mispricing live             Run real settlement-value execution (guarded)
  mispricing live-status      Show real account and strategy allocation
  mispricing portfolio-status Show the live paper portfolio balance

  trade-tape [backtest]       Run the exact-timestamp holdout backtest
  trade-tape train            Train the local win-expectancy model
  trade-tape build-dataset    Rebuild state updates and processed tape
  trade-tape tune             Tune the strategy policy
  trade-tape pipeline         Train, build, tune, then backtest
  trade-tape paper            Continuously run today's and future slates
  trade-tape portfolio-status Show the live paper portfolio balance

Aliases: trade_tape, portable, and portable-trade-tape select trade-tape.

Examples:
  docker run --rm IMAGE mispricing backtest
  docker run --rm IMAGE trade-tape backtest
  docker run --rm IMAGE mispricing paper --input /data/decisions.jsonl
  docker run --rm -e MLB_GAME_PK=... -e KALSHI_MARKET_TICKER=... \
    -e ALLOW_UNVALIDATED_MISPRICING=1 IMAGE mispricing live-paper
  docker run --rm -v "$PWD/data:/app/data" IMAGE setup-data both
  docker run --rm IMAGE paper-both --date YYYY-MM-DD
EOF
}

run_mispricing() {
    operation="${1:-backtest}"
    if [ "$#" -gt 0 ]; then shift; fi
    case "$operation" in
        backtest|evaluate)
            exec python -m settlement_value_strategy.backtest "$@"
            ;;
        train)
            exec python -m settlement_value_strategy.train "$@"
            ;;
        prepare)
            exec python -m settlement_value_strategy.prepare_data "$@"
            ;;
        pipeline)
            python -m settlement_value_strategy.prepare_data "$@"
            python -m settlement_value_strategy.train
            exec python -m settlement_value_strategy.backtest
            ;;
        paper)
            exec python -m settlement_value_strategy.paper_trader "$@"
            ;;
        live-paper|live_paper)
            exec python -m settlement_value_strategy.live_paper_trader \
                --continuous "$@"
            ;;
        live)
            exec python -m settlement_value_strategy.live_paper_trader \
                --all-games "$@"
            ;;
        live-status|live_status)
            exec python -m settlement_value_strategy.live_execution \
                account-status "$@"
            ;;
        portfolio-status|portfolio_status|status)
            exec python -m settlement_value_strategy.live_paper_trader \
                --portfolio-status "$@"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            echo "Unknown mispricing operation: $operation" >&2
            usage >&2
            exit 2
            ;;
    esac
}

run_trade_tape() {
    operation="${1:-backtest}"
    if [ "$#" -gt 0 ]; then shift; fi
    cd "$APP_ROOT/hit_reversion_strategy"
    case "$operation" in
        backtest|evaluate)
            exec python scripts/backtest.py "$@"
            ;;
        train)
            exec python scripts/train_win_model.py "$@"
            ;;
        build-dataset|build_dataset|prepare)
            exec python scripts/build_dataset.py "$@"
            ;;
        tune)
            exec python scripts/tune.py "$@"
            ;;
        pipeline)
            python scripts/train_win_model.py "$@"
            python scripts/build_dataset.py
            python scripts/tune.py
            exec python scripts/backtest.py
            ;;
        paper|live-paper|live_paper)
            exec python scripts/paper_trade.py --continuous "$@"
            ;;
        portfolio-status|portfolio_status|status)
            exec python scripts/paper_trade.py --portfolio-status "$@"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            echo "Unknown trade-tape operation: $operation" >&2
            usage >&2
            exit 2
            ;;
    esac
}

strategy="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi

case "$strategy" in
    paper-both|paper_both|combined-paper)
        exec python "$APP_ROOT/combined_paper.py" "$@"
        ;;
    setup-data|setup_data)
        exec python "$APP_ROOT/setup_data.py" "$@"
        ;;
    mispricing)
        run_mispricing "$@"
        ;;
    trade-tape|trade_tape|portable|portable-trade-tape)
        run_trade_tape "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown strategy: $strategy" >&2
        usage >&2
        exit 2
        ;;
esac
