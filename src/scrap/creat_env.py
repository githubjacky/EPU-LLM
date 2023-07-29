
from pathlib import Path
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--email",
        type=str,
        help="email address to login BigKinds",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="password to login BigKinds",
    )
    args = parser.parse_args()
    return args


def write_env(email: str, password: str):
    file = Path('./src/scrap/.env')
    if not file.is_file():
        file.touch()
    with open(file, 'w') as f:
        f.write(f"EMAIL={email}\n")
        f.write(f"PASSWORD={password}")


def main(args):
    write_env(args.email, args.password)


if __name__ == "__main__":
    main(parse_args())
