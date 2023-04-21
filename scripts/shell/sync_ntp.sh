#!/bin/bash
HOST=pi@10.233.233.3

ssh $HOST "sudo ntpd -s -d"
ssh $HOST "ntpstat"
