# CFT測定用のJulia環境

元測定のProject・ManifestとTNRKitソースをコピーしています。実行検証にはJulia 1.13.2を使用します。Manifest内のローカル依存はこの環境フォルダ内で解決します。

図フォルダから、初回のみ依存パッケージを準備します（未取得パッケージがあればネットワーク接続が必要です）。

```sh
julia --project=environment -e 'using Pkg; Pkg.instantiate()'
```

測定は `../scripts/measure_cft.py` から呼び出します。TNRKitソース、環境定義、測定入口のハッシュは `../measurement_environment.json` に記録しています。環境を変更した場合は新しい測定として来歴と検証結果を記録してください。
