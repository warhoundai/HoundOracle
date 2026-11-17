FROM python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
WORKDIR /opt/oracle
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[api,mcp]' && useradd --uid 10001 --create-home oracle && mkdir /data && chown oracle:oracle /data
USER 10001
VOLUME /data
EXPOSE 8083
ENTRYPOINT ["python", "-m", "houndoracle", "--database", "/data/oracle.sqlite"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8083"]
