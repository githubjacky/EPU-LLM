from pathlib import Path
from argparse import ArgumentParser, Namespace



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="The key to access OpenAI API",
    )
    args = parser.parse_args()
    return args


def main(args):
    file = Path('.env')
    if not file.is_file():
        file.touch()
        with open(file, 'w') as f:
            f.write(f"OPENAI_API_KEY={args.openai_api_key}")

if __name__ == "__main__":
    main(parse_args())
