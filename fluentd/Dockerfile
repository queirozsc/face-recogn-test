# fluentd/Dockerfile
FROM fluent/fluentd:v1.2
# RUN apt-get update && apt-get install -y --no-install-recommends ruby-dev gcc
RUN gem install fluent-plugin-elasticsearch fluent-plugin-out-solr --no-rdoc --no-ri 
COPY fluent.conf /fluentd/etc/fluent.conf
