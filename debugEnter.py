from src.bin.train import run

if __name__ == '__main__':
    # run(
    #     model_name="D2D",
    #     config_path="./configs/d2d-small_tvsub_zh2en.yaml",
    #     log_path="./log/"
    # )
    run(
        model_name="D2D",
        config_path="./configs/d2d-small_tvsub_zh2en.yaml",
        log_path="./log/"
    )