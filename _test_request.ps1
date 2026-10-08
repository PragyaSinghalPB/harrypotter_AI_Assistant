$body = @{question = 'Who is Harry Potter?'} | ConvertTo-Json
$response = Invoke-RestMethod -Uri 'http://127.0.0.1:5001/ask' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 60
$response | ConvertTo-Json -Depth 5
