# fluentd/Dockerfile
FROM fluent/fluentd:v0.14-debian
RUN gem install fluent-plugin-elasticsearch --no-rdoc --no-ri
COPY fluent.conf /fluentd/etc/fluent.conf