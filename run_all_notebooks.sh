#!/bin/zsh
# 指定フォルダ配下の *.ipynb を一括実行し、結果(成功/失敗)をまとめて表示する。
#
# 使い方:
#   ./run_all_notebooks.sh [対象フォルダ (省略時はカレント)] [Pythonパス] [1セルあたりのタイムアウト秒]
#
# 例:
#   ./run_all_notebooks.sh .                                 # ml-jitumu-code 配下を全部実行
#   ./run_all_notebooks.sh 01_credit                             # 01_credit フォルダだけ実行
#   ./run_all_notebooks.sh . ~/.pyenv/versions/3.10.17/bin/python 3600
#
# 前提: 対象の Python 環境に nbconvert, ipykernel がインストール済みであること。
#   未インストールの場合は先に以下を実行:
#   <python> -m pip install -q nbconvert nbformat jupyter_client ipykernel
#
# 実行結果は notebook ファイル自体に書き戻される (--inplace)。
# 詳細ログは $LOG に、失敗一覧は最後にターミナルへ出力される。

set -u

TARGET_DIR="${1:-.}"
PY="${2:-$(command -v python3)}"
TIMEOUT="${3:-1800}"
LOG="/tmp/nb_run_$(date +%Y%m%d_%H%M%S).log"
PY="$($PY -c 'import sys; print(sys.executable)')" || exit 1
RUNTIME_ROOT=$(mktemp -d) || exit 1
trap 'rm -rf "$RUNTIME_ROOT"' EXIT
export JUPYTER_CONFIG_DIR="$RUNTIME_ROOT/jupyter-config"
export JUPYTER_RUNTIME_DIR="$RUNTIME_ROOT/jupyter-runtime"
export IPYTHONDIR="$RUNTIME_ROOT/ipython"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"
export JUPYTER_PATH="$RUNTIME_ROOT/kernel/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
"$PY" -m ipykernel install --prefix "$RUNTIME_ROOT/kernel" --name ml-code-validation || exit 1

echo "対象フォルダ: $TARGET_DIR"
echo "Python: $PY"
echo "タイムアウト(秒/セル): $TIMEOUT"
echo "ログ: $LOG"
: > "$LOG"

typeset -a NOTEBOOKS
NOTEBOOKS=("${(@f)$(find "$TARGET_DIR" -type d \( -name '.ipynb_checkpoints' -o -name 'tmp' \) -prune -o -type f -name '*.ipynb' -print | sort)}")

typeset -a OK_LIST FAIL_LIST

for nb in "${NOTEBOOKS[@]}"; do
  [[ -z "$nb" ]] && continue
  echo "=== START: $nb ===" | tee -a "$LOG"
  start=$(date +%s)
  if PYTHONWARNINGS="error::RuntimeWarning${PYTHONWARNINGS:+,$PYTHONWARNINGS}" "$PY" -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout="$TIMEOUT" \
      --ExecutePreprocessor.kernel_name=ml-code-validation \
      "$nb" >> "$LOG" 2>&1; then
    end=$(date +%s)
    echo "=== OK: $nb ($((end-start))s) ===" | tee -a "$LOG"
    OK_LIST+=("$nb")
  else
    end=$(date +%s)
    echo "=== FAIL: $nb ($((end-start))s) ===" | tee -a "$LOG"
    FAIL_LIST+=("$nb")
  fi
done

echo ""
echo "########## 結果サマリ ##########"
echo "成功 (${#OK_LIST[@]}件):"
for nb in "${OK_LIST[@]}"; do echo "  OK   $nb"; done
echo ""
echo "失敗 (${#FAIL_LIST[@]}件):"
for nb in "${FAIL_LIST[@]}"; do echo "  FAIL $nb"; done
echo ""
echo "詳細ログ: $LOG"

if (( ${#FAIL_LIST[@]} > 0 )); then
  exit 1
fi
exit 0
