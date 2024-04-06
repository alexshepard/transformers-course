from pathlib import Path

from transformers import BertConfig, BertModel


def main():
    config = BertConfig()

    # untrained model
    model = BertConfig(config)
    print(config)

    # pretrained model
    model = BertModel.from_pretrained("bert-base-cased")
    print(model.config)

    # saving modes
    save_dir = Path(__file__).parents[1] / "models"
    model.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
