"""Evaluate with the VLFM object-navigation base (NAV_BASE=vlfm).

Same command line, configuration handling, datasets, logs and result files as
PSG_Nav.py: this runs PSG_Nav.main() with nav_bases.vlfm_agent.VLFMNavAgent in place
of PSG_Nav_Agent, in this process only. PSG_Nav.py itself is not modified, so the
default PSG-Nav base (python PSG_Nav.py ...) is byte-for-byte the code it was.

ENABLE_INSTANCE_CONFIRMATION=false: original VLFM. true (with
ENABLE_CANDIDATE_APPROACH=false): VLFM + AVS. The VLFM model servers must be running
(scripts/launch_vlfm_servers.sh).
"""
import os


def main():
    os.environ['NAV_BASE'] = 'vlfm'
    import PSG_Nav
    from nav_bases.vlfm_agent import VLFMNavAgent
    PSG_Nav.PSG_Nav_Agent = VLFMNavAgent
    PSG_Nav.main()


if __name__ == '__main__':
    main()
