const videoInput = document.getElementById('videoInput');
        const depthMap = document.getElementById('depthMap');
        const log = document.getElementById('log');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoInput.srcObject = stream;
                processVideo();
            });

        function processVideo() {
            const canvas = document.createElement('canvas');
            canvas.width = videoInput.width;
            canvas.height = videoInput.height;
            const ctx = canvas.getContext('2d');

            setInterval(() => {
                ctx.drawImage(videoInput, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg');

                fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                })
                .then(response => response.json())
                .then(data => {
                    depthMap.src = 'data:image/jpeg;base64,' + data.depth_map;
                    log.value += 'Updated at: ' + new Date().toLocaleTimeString() + '\n';
                });
            }, 5000);  // Update every 5 seconds
        }