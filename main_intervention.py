import argparse
import yaml
import os
from train.train_cb_vaegan import main as train_cb_vaegan
from train.train_vaegan import main as train_vaegan

from intervene.simple_intervention_generation import main as simple_intervention_generation

#WANDB CONFIGURATIONS FOR CLUSTER
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # We only specify the yaml file from argparse and handle rest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dataset", default="color_mnist", help="benchmark dataset"
    )

    args = parser.parse_args()
    args.config_file = "./config/cb_vaegan/" + args.dataset + "_intervention.yaml"

    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")

    simple_intervention_generation(config)


if __name__ == "__main__":
    main()