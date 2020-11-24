import logging
from obiwan.batch import TaskManager
from obiwan import setup_logging

setup_logging(logging.DEBUG)

def test_batch():
    with TaskManager(ntasks=1) as tm:
        lit = list(range(10))
        li = []
        for i in tm.iterate(lit):
            li.append(i)
        assert li == lit
        li = tm.map(lambda i: i+1,lit)
        assert li == list(range(1,len(lit)+1))
