
# Installation

## Anaconda

You'll need Anaconda, so if you don't have it yet, just install it now.
Follow [the anaconda installation instructions](https://conda.io/docs/user-guide/install/linux.html).
We propose to use the most recent version.

    wget https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh
    bash ./Anaconda3-5.0.0.1-Linux-x86_64.sh

## digicamtoy

We propose to have a tidy place and clone `digicamtoy` into a folder `ctasoft/`

    mkdir ctasoft
    cd ctasoft
    git clone https://github.com/cta-sst-1m/digicamtoy

To not mix up your anaconda root environment with digicamtoy, we propose
to make a so called environment, with all the dependencies in it.

    conda env create -f digicamtoy/environment.yml
    source activate digicamtoy

**Please Note**: When working with digicamtoy, please always make sure you are really using the `digicamtoy` environment. After `source activate digicamtoy`
your prompt should look similar to this this:

    (digicamtoy) username@host:~/ctasoft$

    pip install -e digicampipe

Finally change into the digicampipe directory.

    cd digicampipe