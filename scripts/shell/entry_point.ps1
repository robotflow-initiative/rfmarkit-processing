python -m

.probe --config=./config/config.json | powershell
Start-Process python -ArgumentList "-m", "tcpbroker.main", "--config=./config/config.json", "-P"
Start-Process python -ArgumentList "-m", "tcpbroker.main", "--config=./config/config.json", "--easy"
Start-Process python -ArgumentList "src/model.py", "--config=./config/config.json"