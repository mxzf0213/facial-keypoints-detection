console.log('Loading Model...');
async function loadModel() {
    const model = await tf.loadLayersModel('./model/model.json')
    return model
}

const videoWidth = 600
const videoHeight = 500

const new_width = 96
const new_height = 96

const color = 'aqua'

function isAndroid() {
    return /Android/i.test(navigator.userAgent);
}

function isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
    return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;

    const mobile = isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: mobile ? undefined : videoWidth,
            height: mobile ? undefined : videoHeight,
        },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();

    return video;
}

function preprocess(imgData) {
    //document.getElementById('txt').innerHTML = tf.fromPixels(imgData).toFloat()
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(imgData).toFloat()
        // Normalize the image 
        const normalized = tensor.div(tf.scalar(255.0))
        const resized = tf.image.resizeBilinear(normalized, [new_width, new_height])
        const sliced = resized.slice([0, 0, 1], [new_width, new_height, 1])
        const batched = sliced.expandDims(0)
        return batched
    })
}

async function predict(net, ctx) {
    imgData = ctx.getImageData(0, 0, video.width, video.height)
    const pred = net.predict(preprocess(imgData)).dataSync()
    return pred
}

async function drawKeyPoints(features, ctx) {
    for (var i = 0; i < features.length; i += 2) {
        features[i] = (features[i] * 48 + 48) / 96 * videoWidth;
        features[i + 1] = (features[i + 1] * 48 + 48) / 96 * videoHeight;
        ctx.beginPath();
        ctx.arc(features[i], features[i + 1], 3, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
    }
}
function detectPoseInRealTime(video, net) {
    const canvas = document.getElementById('output');
    const ctx = canvas.getContext('2d');

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    async function poseDetectionFrame() {
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        let features = []
        features = await predict(net, ctx)
        await drawKeyPoints(features, ctx)
        ctx.restore();
        requestAnimationFrame(poseDetectionFrame);
    }
    poseDetectionFrame();
}

async function bindPage() {
    const faceModel = await loadModel()

    console.log('Model loaded!')
    faceModel.summary()

    document.getElementById('main').style.display = 'block';

    let video;

    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = 'this browser does not support video capture,' +
            'or this device does not have a camera';
        info.style.display = 'block';
        throw e;
    }

    detectPoseInRealTime(video, faceModel);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

bindPage();