# Live violence monitor

This tutorial shows how to setup a system to monitor the level of violence directed to a set of Twitter profiles on the hour.

## Prerequisites

### Twitter API bearer token

The Python code employed in this tutorial makes use of the Twitter API v2 ("mentions" endpoint), which can only be accessed using a private token
(the bearer token) associated to a Twitter developer account. Follow the instructions below to get one for yourself.

1. Create a twitter account at: <https://twitter.com/i/flow/signup>.
2. Become a developer at: <https://developer.twitter.com/en/portal/petition/essential/basic-info>.
3. Create a new project by following the instructions at: <https://developer.twitter.com/en/docs/projects/overview>.
4. Follow the same instructions above to create an App for your project.
5. When you create your Twitter App, you will be presented with your API Key and Secret, along with a Bearer Token. You should save these locally as they are only showed once.

You should save the tokens in the file at [../../keys/twitter_violentometro.json](../../keys/twitter_violentometro.json), in the following format:

    {
        "api_key": "<your twitter API key>",
        "api_secret_key": "<your twitter API secret key>",
        "bearer_token": "<your twitter API bearer key>",
        "access_token": "<your twitter API access token>",
        "access_token_secret": "<your twitter API token secret>"
    }

For this project, you only need to put your Bearer token (also known as app-only token) in the JSON above.


### Tensorflow and other python packages

This project requires tensorflow for rating the tweets and works best when using GPU. To install Tensorflow with GPU support, follow the
instructions in this link: <https://www.tensorflow.org/install/pip>. Once installed, it is recommended to install the remaining necessary
packages for the project in the same environment (Tensorflow recommends using miniconda) with pip. For that, go to the root folder of
this project and run (assuming you have pip installed):

    conda activate tf
    pip install -r requirements.txt
    conda deactivate


## Running the monitor


### Configutaring



### Running

On Unix-like systems (e.g. Linux and Mac) and when logged on a server through ssh, you may run the monitoring system
by executing the command `./run_monitor_on_server.sh <run tag>` in this folder, where `<run tag>` is replaced by some
identifier of the current monitor run, like the date (e.g. 2023-03-01). This shell script does three things:

1. Starts the `tf` conda environment;
2. Starts 3 independent processes (tweet capture, tweet rating for violence and data analysis) with [nohup](https://en.wikipedia.org/wiki/Nohup); and
3. Redirect their outputs to logs located at [tweets/logs/](tweets/logs/), identified by the `<run tag>`.

With nohup, you may logout from the server while the three processes responsible for the violence monitoring will stay running.
