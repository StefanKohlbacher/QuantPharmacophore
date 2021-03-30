FROM centos:7

COPY . /qphar

WORKDIR /qphar

ENTRYPOINT ["tail"]

CMD ["-f", "/dev/null"]
