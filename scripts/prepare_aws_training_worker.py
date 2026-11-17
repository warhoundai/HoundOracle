"""Prepare inspectable CPU training worker userdata with independent termination timer."""

import json
from pathlib import Path


def main():
    public = Path("research/experiments/campaign-19")
    inputs = json.loads((public / "training-inputs.json").read_text())
    sha = inputs["archive_sha256"]
    boot = """#!/bin/bash
set -euo pipefail
systemd-run --unit=houndoracle-training19-watchdog --on-active=45m /usr/sbin/shutdown -h now
mkdir -p /opt/houndoracle
finish() {
 /opt/houndoracle/venv/bin/python - <<'LOG' || true
import boto3
boto3.client('s3',region_name='us-east-1').upload_file('/var/log/cloud-init-output.log','houndoracle-research-751446514671-20261116','training19/output/bootstrap.log')
LOG
 /usr/sbin/shutdown -h now
}
trap finish EXIT
cat > /opt/houndoracle/start.sh <<'START'
#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip
python3 -m venv /opt/houndoracle/venv
/opt/houndoracle/venv/bin/pip install --disable-pip-version-check --no-cache-dir boto3==1.44.3
/opt/houndoracle/venv/bin/python - <<'PROGRESS'
import boto3,json
boto3.client('s3',region_name='us-east-1').put_object(Bucket='houndoracle-research-751446514671-20261116',Key='training19/output/progress.json',Body=json.dumps({'state':'INSTALLING_TRAINING_PACKAGES'}).encode())
PROGRESS
/opt/houndoracle/venv/bin/pip install --disable-pip-version-check --no-cache-dir --index-url=https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
/opt/houndoracle/venv/bin/pip install --disable-pip-version-check --no-cache-dir transformers==5.23.0 numpy==2.5.4 safetensors==0.8.0 onnx==1.23.2 onnxruntime==1.32.0
/opt/houndoracle/venv/bin/python - <<'DOWNLOAD'
import boto3,hashlib,tarfile
from pathlib import Path
root=Path('/opt/houndoracle')
boto3.client('s3',region_name='us-east-1').download_file('houndoracle-research-751446514671-20261116','training19/input/bundle.tar.gz',str(root/'bundle.tar.gz'))
assert hashlib.sha256((root/'bundle.tar.gz').read_bytes()).hexdigest()==__EXPECTED_SHA__
with tarfile.open(root/'bundle.tar.gz') as archive:archive.extractall(root,filter='data')
DOWNLOAD
cd /opt/houndoracle
timeout --signal=TERM --kill-after=120s 2100s /opt/houndoracle/venv/bin/python -u train.py > training.log 2>&1
START
chmod 700 /opt/houndoracle/start.sh
timeout --signal=TERM --kill-after=120s 2400s /bin/bash /opt/houndoracle/start.sh
""".replace("__EXPECTED_SHA__", repr(sha))
    target = Path("/workspace/houndoracle-data/campaign-19/userdata.sh")
    target.write_text(boot)
    record = {
        "region": "us-east-1",
        "instance_type": "c7i.2xlarge",
        "verified_on_demand_compute_usd_per_hour": 0.357,
        "compute_price_source": "AWS Pricing GetProducts Linux shared us-east-1 2026-11-16",
        "reserved_usd": 10,
        "maximum_instances": 1,
        "ami": "ami-08b275ac6cc5dbefb",
        "verified_image_owner": "099720109477",
        "vcpus": 8,
        "regional_standard_vcpu_quota": 8,
        "observed_other_running_or_pending_instances": 0,
        "regional_gpu_vcpu_quota": 0,
        "role": "houndoracle-training19-20261116",
        "security_group": "sg-00eb6dfa343cea1bc",
        "subnet": "subnet-0a2d6263cf032fa60",
        "no_inbound_rules": True,
        "no_ssh_key": True,
        "root_volume_gib": 20,
        "root_volume_encrypted": True,
        "root_volume_delete_on_termination": True,
        "instance_shutdown_behavior": "terminate",
        "imds_tokens": "required",
        "watchdog_minutes": 45,
        "whole_worker_timeout_seconds": 2400,
        "training_loop_timeout_seconds": 1800,
        "training_process_timeout_seconds": 2100,
        "billing_and_credits": "Credit balance and final invoice unverified; user authorization covers pay-as-you-go within campaign ceiling. Compute estimate excludes EBS, IPv4, storage, transfer and request charges; ten-dollar reservation retains margin. Watchdog requires successful bootstrap; externally monitor and terminate task-owned instance.",
        "source_sha256": inputs["training_script_sha256"],
        "status": "PREPARED_NOT_LAUNCHED",
    }
    (public / "aws-preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
