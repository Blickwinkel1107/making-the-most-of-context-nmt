from src.bin.train import run

if __name__ == '__main__':
    run(
        model_name="docnmt",
        reload=True,
        config_path="./configs/transformer_tvsub_zh2en.yaml",
        log_path="./log/",
        saveto="./save/"
    )