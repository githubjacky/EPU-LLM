FROM python:3.11.4-bookworm
WORKDIR /epu_denoise


# no need to create virtual environment since the docker containr is already is
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

ENV PATH="$POETRY_HOME/bin:$PATH"

# dependencies to install poetry and set up ODBC
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.6.0
COPY pyproject.toml ./
RUN poetry install
EXPOSE 8888


CMD ["bash"]
