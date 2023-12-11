ps -ef | grep "manage.py" | awk {'print "kill -9 " $2'} | sh -x
