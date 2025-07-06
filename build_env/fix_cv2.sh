tar -zxvf build_env/zlib-1.2.9.tar.gz -C /tmp
cd /tmp/zlib-1.2.9
./configure
make & make install
ln -s -f /usr/local/lib/libz.so.1.2.9 /opt/hadoop-2.10.sdi-080-client-batch/lib/native/libz.so.1