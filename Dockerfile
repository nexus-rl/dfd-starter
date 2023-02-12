FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update

RUN apt-get install -y python3 python3-pip build-essential curl clang libclang-dev git && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements-docker.txt /app

RUN python3 -m pip install maturin==0.14.12
RUN python3 -m pip install -r requirements-docker.txt

# copy the code over now - allows us to change code w/o having to wait for all
# the requirements to reinstall
COPY . /app

EXPOSE 10250

CMD python3 run_server.py train --env RocketSim-v0 --log_to_wandb --wandb_project RocketSim-v0 --wandb_run_name rocketsim_test-$(date -u -Iminutes | sed 's/\+.*//g') --vbn_buffer_size 128 --no-normalize_obs --bind_address 0.0.0.0 --bind_port 10250

