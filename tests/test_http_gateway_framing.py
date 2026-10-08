import socket
import threading
import pytest
from shard.http_gateway import AuthRegistry, Gateway
from shard.service_queue import TenantLimits

KEY='test-api-key-not-a-secret-123456'


class Backend:
    model_id='synthetic-model'
    calls=0
    def ready(self):return True,'synthetic_test_warmup'
    def stats(self):return {}
    def prepare(self,body):
        self.calls+=1
        return body,1,1,2
    def execute(self,job,emit,cancel_check):
        emit(1)
        return {'tokens':[1],'text':'x','finish_reason':'stop','proof':{'verified':True,'scope':'synthetic_test'},'recovery':{}}
    def decode(self,tokens):return 'x'
    def abort(self):pass


@pytest.fixture
def server():
    backend=Backend()
    gateway=Gateway(backend,AuthRegistry({KEY:'tenant'},{'tenant':TenantLimits()}))
    http=gateway.server('127.0.0.1',0)
    worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
    yield http,backend
    http.shutdown();http.server_close();gateway.shutdown();worker.join(2)


@pytest.mark.parametrize('headers,body',[
    ('Content-Length: 2\r\nContent-Length: 2\r\n',b'{}'),
    ('Content-Length: 2\r\nTransfer-Encoding: chunked\r\n',b'{}'),
    ('Content-Length: +2\r\n',b'{}'),
    ('Content-Length: 13\r\n',b'{"x":1,"x":2}'),
    ('Content-Length: 9\r\n',b'{"x":NaN}')])
def test_ambiguous_frames_and_json_rejected_before_admission(server,headers,body):
    http,backend=server
    with socket.create_connection(http.server_address,timeout=3) as client:
        request=('POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nAuthorization: Bearer '+KEY+'\r\n'+headers+'Connection: close\r\n\r\n').encode()+body
        client.sendall(request)
        response=client.recv(4096)
        assert b'400 Bad Request' in response
    assert backend.calls==0
