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
