/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';

import SpotifyWebApi from 'spotify-web-api-js';

import { drawBoundingBox, drawKeypoints, drawSkeleton, isMobile, toggleLoadingUI, tryResNetButtonName, tryResNetButtonText, updateTryResNetButtonDatGuiCss } from './demo_util';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

let currentState = "";

const spotifyApi = new SpotifyWebApi();

const songLookup = {
    "YMCA": {
        uri: 'spotify:track:7Cp69rNBwU0gaFT8zxExlE',
        position: 58 * 1000
    },
    "Baby Shark": {
        uri: 'spotify:track:5ygDXis42ncn6kYG14lEVG',
        position: 4 * 1000
    },
    "Disco": {
        uri: 'spotify:track:7qK3JFriCqLorQivsJYG2X',
        position: 0
    },
    "Thriller": {
        uri: 'spotify:track:7azo4rpSUh8nXgtonC6Pkq',
        position: 90.5 * 1000
    }
}

async function updateState(newState) {
    if (currentState === newState) {
        return;
    }

    console.log('Updating State:', newState);

    const currentDevice = getSelectedDevice();

    if (newState in songLookup) {
        var song = songLookup[newState];
        try {
            await spotifyApi.play({
                device_id: currentDevice,
                uris: [song.uri]
            });

            if (song.position > 0) {
                await spotifyApi.seek(song.position);
            }
        }
        catch (ex) {
            if (ex.status === 401) {
                authorize();
            }
        }
    }
    else {
        console.error("Could not find the song for the state:", newState)
    }

    currentState = newState;
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

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 500;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 250;

const guiState = {
    algorithm: 'multi-pose',
    input: {
        architecture: 'MobileNetV1',
        outputStride: defaultMobileNetStride,
        inputResolution: defaultMobileNetInputResolution,
        multiplier: defaultMobileNetMultiplier,
        quantBytes: defaultQuantBytes
    },
    singlePoseDetection: {
        minPoseConfidence: 0.1,
        minPartConfidence: 0.5,
    },
    multiPoseDetection: {
        maxPoseDetections: 5,
        minPoseConfidence: 0.15,
        minPartConfidence: 0.1,
        nmsRadius: 30.0,
    },
    output: {
        showVideo: true,
        showSkeleton: true,
        showPoints: true,
        showBoundingBox: false,
    },
    net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
    guiState.net = net;

    if (cameras.length > 0) {
        guiState.camera = cameras[0].deviceId;
    }

    const gui = new dat.GUI({ width: 300 });
    gui.close();

    let architectureController = null;
    guiState[tryResNetButtonName] = function () {
        architectureController.setValue('ResNet50')
    };
    gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
    updateTryResNetButtonDatGuiCss();

    // The single-pose algorithm is faster and simpler but requires only one
    // person to be in the frame or results will be innaccurate. Multi-pose works
    // for more than 1 person
    const algorithmController =
        gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

    // The input parameters have the most effect on accuracy and speed of the
    // network
    let input = gui.addFolder('Input');
    // Architecture: there are a few PoseNet models varying in size and
    // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
    // fastest, but least accurate.
    architectureController =
        input.add(guiState.input, 'architecture', ['MobileNetV1', 'ResNet50']);
    guiState.architecture = guiState.input.architecture;
    // Input resolution:  Internally, this parameter affects the height and width
    // of the layers in the neural network. The higher the value of the input
    // resolution the better the accuracy but slower the speed.
    let inputResolutionController = null;
    function updateGuiInputResolution(
        inputResolution,
        inputResolutionArray,
    ) {
        if (inputResolutionController) {
            inputResolutionController.remove();
        }
        guiState.inputResolution = inputResolution;
        guiState.input.inputResolution = inputResolution;
        inputResolutionController =
            input.add(guiState.input, 'inputResolution', inputResolutionArray);
        inputResolutionController.onChange(function (inputResolution) {
            guiState.changeToInputResolution = inputResolution;
        });
    }

    // Output stride:  Internally, this parameter affects the height and width of
    // the layers in the neural network. The lower the value of the output stride
    // the higher the accuracy but slower the speed, the higher the value the
    // faster the speed but lower the accuracy.
    let outputStrideController = null;
    function updateGuiOutputStride(outputStride, outputStrideArray) {
        if (outputStrideController) {
            outputStrideController.remove();
        }
        guiState.outputStride = outputStride;
        guiState.input.outputStride = outputStride;
        outputStrideController =
            input.add(guiState.input, 'outputStride', outputStrideArray);
        outputStrideController.onChange(function (outputStride) {
            guiState.changeToOutputStride = outputStride;
        });
    }

    // Multiplier: this parameter affects the number of feature map channels in
    // the MobileNet. The higher the value, the higher the accuracy but slower the
    // speed, the lower the value the faster the speed but lower the accuracy.
    let multiplierController = null;
    function updateGuiMultiplier(multiplier, multiplierArray) {
        if (multiplierController) {
            multiplierController.remove();
        }
        guiState.multiplier = multiplier;
        guiState.input.multiplier = multiplier;
        multiplierController =
            input.add(guiState.input, 'multiplier', multiplierArray);
        multiplierController.onChange(function (multiplier) {
            guiState.changeToMultiplier = multiplier;
        });
    }

    // QuantBytes: this parameter affects weight quantization in the ResNet50
    // model. The available options are 1 byte, 2 bytes, and 4 bytes. The higher
    // the value, the larger the model size and thus the longer the loading time,
    // the lower the value, the shorter the loading time but lower the accuracy.
    let quantBytesController = null;
    function updateGuiQuantBytes(quantBytes, quantBytesArray) {
        if (quantBytesController) {
            quantBytesController.remove();
        }
        guiState.quantBytes = +quantBytes;
        guiState.input.quantBytes = +quantBytes;
        quantBytesController =
            input.add(guiState.input, 'quantBytes', quantBytesArray);
        quantBytesController.onChange(function (quantBytes) {
            guiState.changeToQuantBytes = +quantBytes;
        });
    }

    function updateGui() {
        if (guiState.input.architecture === 'MobileNetV1') {
            updateGuiInputResolution(
                defaultMobileNetInputResolution,
                [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]);
            updateGuiOutputStride(defaultMobileNetStride, [8, 16]);
            updateGuiMultiplier(defaultMobileNetMultiplier, [0.50, 0.75, 1.0]);
        } else {  // guiState.input.architecture === "ResNet50"
            updateGuiInputResolution(
                defaultResNetInputResolution,
                [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]);
            updateGuiOutputStride(defaultResNetStride, [32, 16]);
            updateGuiMultiplier(defaultResNetMultiplier, [1.0]);
        }
        updateGuiQuantBytes(defaultQuantBytes, [1, 2, 4]);
    }

    updateGui();
    input.open();
    // Pose confidence: the overall confidence in the estimation of a person's
    // pose (i.e. a person detected in a frame)
    // Min part confidence: the confidence that a particular estimated keypoint
    // position is accurate (i.e. the elbow's position)
    let single = gui.addFolder('Single Pose Detection');
    single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
    single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

    let multi = gui.addFolder('Multi Pose Detection');
    multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
        .min(1)
        .max(20)
        .step(1);
    multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
    multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
    // nms Radius: controls the minimum distance between poses that are returned
    // defaults to 20, which is probably fine for most use cases
    multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
    multi.open();

    let output = gui.addFolder('Output');
    output.add(guiState.output, 'showVideo');
    output.add(guiState.output, 'showSkeleton');
    output.add(guiState.output, 'showPoints');
    output.add(guiState.output, 'showBoundingBox');
    output.open();


    architectureController.onChange(function (architecture) {
        // if architecture is ResNet50, then show ResNet50 options
        updateGui();
        guiState.changeToArchitecture = architecture;
    });

    algorithmController.onChange(function (value) {
        switch (guiState.algorithm) {
            case 'single-pose':
                multi.close();
                single.open();
                break;
            case 'multi-pose':
                single.close();
                multi.open();
                break;
        }
    });


}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
    stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
    document.getElementById('main').appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
    const canvas = document.getElementById('output');
    const ctx = canvas.getContext('2d');

    // since images are being fed from a webcam, we want to feed in the
    // original image and then just flip the keypoints' x coordinates. If instead
    // we flip the image, then correcting left-right keypoint pairs requires a
    // permutation on all the keypoints.
    const flipPoseHorizontal = true;

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    async function poseDetectionFrame() {
        if (guiState.changeToArchitecture) {
            // Important to purge variables and free up GPU memory
            guiState.net.dispose();
            toggleLoadingUI(true);
            guiState.net = await posenet.load({
                architecture: guiState.changeToArchitecture,
                outputStride: guiState.outputStride,
                inputResolution: guiState.inputResolution,
                multiplier: guiState.multiplier,
            });
            toggleLoadingUI(false);
            guiState.architecture = guiState.changeToArchitecture;
            guiState.changeToArchitecture = null;
        }

        if (guiState.changeToMultiplier) {
            guiState.net.dispose();
            toggleLoadingUI(true);
            guiState.net = await posenet.load({
                architecture: guiState.architecture,
                outputStride: guiState.outputStride,
                inputResolution: guiState.inputResolution,
                multiplier: +guiState.changeToMultiplier,
                quantBytes: guiState.quantBytes
            });
            toggleLoadingUI(false);
            guiState.multiplier = +guiState.changeToMultiplier;
            guiState.changeToMultiplier = null;
        }

        if (guiState.changeToOutputStride) {
            // Important to purge variables and free up GPU memory
            guiState.net.dispose();
            toggleLoadingUI(true);
            guiState.net = await posenet.load({
                architecture: guiState.architecture,
                outputStride: +guiState.changeToOutputStride,
                inputResolution: guiState.inputResolution,
                multiplier: guiState.multiplier,
                quantBytes: guiState.quantBytes
            });
            toggleLoadingUI(false);
            guiState.outputStride = +guiState.changeToOutputStride;
            guiState.changeToOutputStride = null;
        }

        if (guiState.changeToInputResolution) {
            // Important to purge variables and free up GPU memory
            guiState.net.dispose();
            toggleLoadingUI(true);
            guiState.net = await posenet.load({
                architecture: guiState.architecture,
                outputStride: guiState.outputStride,
                inputResolution: +guiState.changeToInputResolution,
                multiplier: guiState.multiplier,
                quantBytes: guiState.quantBytes
            });
            toggleLoadingUI(false);
            guiState.inputResolution = +guiState.changeToInputResolution;
            guiState.changeToInputResolution = null;
        }

        if (guiState.changeToQuantBytes) {
            // Important to purge variables and free up GPU memory
            guiState.net.dispose();
            toggleLoadingUI(true);
            guiState.net = await posenet.load({
                architecture: guiState.architecture,
                outputStride: guiState.outputStride,
                inputResolution: guiState.inputResolution,
                multiplier: guiState.multiplier,
                quantBytes: guiState.changeToQuantBytes
            });
            toggleLoadingUI(false);
            guiState.quantBytes = guiState.changeToQuantBytes;
            guiState.changeToQuantBytes = null;
        }

        // Begin monitoring code for frames per second
        stats.begin();

        let poses = [];
        let minPoseConfidence;
        let minPartConfidence;
        switch (guiState.algorithm) {
            case 'single-pose':
                const pose = await guiState.net.estimatePoses(video, {
                    flipHorizontal: flipPoseHorizontal,
                    decodingMethod: 'single-person'
                });
                poses = poses.concat(pose);
                minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
                minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
                break;
            case 'multi-pose':
                let all_poses = await guiState.net.estimatePoses(video, {
                    flipHorizontal: flipPoseHorizontal,
                    decodingMethod: 'multi-person',
                    maxDetections: guiState.multiPoseDetection.maxPoseDetections,
                    scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
                    nmsRadius: guiState.multiPoseDetection.nmsRadius
                });

                poses = poses.concat(all_poses);
                minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
                minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
                break;
        }

        ctx.clearRect(0, 0, videoWidth, videoHeight);

        if (guiState.output.showVideo) {
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-videoWidth, 0);
            ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
            ctx.restore();
        }

        // For each pose (i.e. person) detected in an image, loop through the poses
        // and draw the resulting skeleton and keypoints if over certain confidence
        // scores
        poses.forEach(({ score, keypoints }) => {
            if (score >= minPoseConfidence) {
                if (guiState.output.showPoints) {
                    drawKeypoints(keypoints, minPartConfidence, ctx);
                }
                if (guiState.output.showSkeleton) {
                    drawSkeleton(keypoints, minPartConfidence, ctx);
                }
                if (guiState.output.showBoundingBox) {
                    drawBoundingBox(keypoints, ctx);
                }
            }
        });

        // try to detect the pose for music

        poses.forEach(async ({ score, keypoints }) => {
            if (score >= minPoseConfidence) {
                // console.log(keypoints);

                function isAllConfidentParts(partNames) {
                    return partNames.map(partName => {
                        var keypoint = keypoints.find(kp => kp.part === partName);
                        return keypoint.score;
                    })
                        .every(confidence => {
                            return confidence > minPartConfidence;
                        });
                }

                function getScore(partName) {
                    var keypoint = keypoints.find(kp => kp.part === partName);
                    return keypoint.score;
                }

                function getX(partName) {
                    var keypoint = keypoints.find(kp => kp.part === partName);
                    return keypoint.position.x;
                }

                function getY(partName) {
                    var keypoint = keypoints.find(kp => kp.part === partName);
                    return keypoint.position.y;
                }

                const headParts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar'];

                function headX() {
                    let vals = headParts.filter(partName => {
                        return getScore(partName) > minPartConfidence;
                    })
                        .map(partName => {
                            return getX(partName);
                        });

                    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                    return avg;
                }

                function headY() {
                    let vals = headParts.filter(partName => {
                        return getScore(partName) > minPartConfidence;
                    })
                        .map(partName => {
                            return getY(partName);
                        });

                    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                    return avg;
                }

                // check for hands above heads
                if (isAllConfidentParts(['leftWrist', 'leftElbow', 'leftShoulder', 'rightWrist', 'rightElbow', 'rightShoulder']) &&
                    (
                        getY('leftWrist') < getY('leftElbow') && getY('leftElbow') < getY('leftShoulder') &&
                        getY('rightWrist') < getY('rightElbow') && getY('rightElbow') < getY('rightShoulder')
                    ) &&
                    (
                        (getX('leftWrist') < headX() && getX('rightWrist') > headX()) ||
                        (getX('leftWrist') > headX() && getX('rightWrist') < headX())
                    )
                ) {
                    await updateState('YMCA');
                }
                // check for sharks
                else if (
                    isAllConfidentParts(['leftWrist', 'leftElbow', 'leftShoulder', 'rightWrist', 'rightElbow', 'rightShoulder']) &&
                    (
                        (getY('leftWrist') < getY('leftElbow') && getY('leftElbow') < getY('leftShoulder') &&
                            getY('rightWrist') > getY('rightElbow') && getY('rightElbow') > getY('rightShoulder'))
                        ||
                        ((getY('leftWrist') > getY('leftElbow') && getY('leftElbow') > getY('leftShoulder') &&
                            getY('rightWrist') < getY('rightElbow') && getY('rightElbow') < getY('rightShoulder')))
                    ) &&
                    (
                        (getX('leftWrist') < headX() && getX('rightWrist') < headX()) ||
                        (getX('leftWrist') > headX() && getX('rightWrist') > headX())
                    )
                ) {
                    await updateState('Baby Shark');
                }
                // check for disco
                else if (
                    isAllConfidentParts(['leftWrist', 'leftElbow', 'leftShoulder', 'rightWrist', 'rightElbow', 'rightShoulder']) &&
                    (
                        (getY('leftWrist') < getY('leftElbow') && getY('leftElbow') < getY('leftShoulder') &&
                            getY('rightWrist') > getY('rightElbow') && getY('rightElbow') > getY('rightShoulder'))
                        ||
                        ((getY('leftWrist') > getY('leftElbow') && getY('leftElbow') > getY('leftShoulder') &&
                            getY('rightWrist') < getY('rightElbow') && getY('rightElbow') < getY('rightShoulder')))
                    ) &&
                    (
                        (getX('leftWrist') < headX() && getX('rightWrist') > headX()) ||
                        (getX('leftWrist') > headX() && getX('rightWrist') < headX())
                    )
                ) {
                    await updateState('Disco');
                }
                // check for thriller
                else if (
                    isAllConfidentParts(['leftWrist', 'leftElbow', 'rightWrist', 'rightElbow']) &&
                    (
                        (getY('leftWrist') < getY('leftElbow') &&
                            getY('rightWrist') < getY('rightElbow'))

                    ) &&
                    (
                        (getX('leftWrist') < headX() && getX('rightWrist') < headX()) ||
                        (getX('leftWrist') > headX() && getX('rightWrist') > headX())
                    )
                ) {
                    await updateState('Thriller');
                }

            }
        });

        // End monitoring code for frames per second
        stats.end();

        requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
    toggleLoadingUI(true);
    const net = await posenet.load({
        architecture: guiState.input.architecture,
        outputStride: guiState.input.outputStride,
        inputResolution: guiState.input.inputResolution,
        multiplier: guiState.input.multiplier,
        quantBytes: guiState.input.quantBytes
    });
    toggleLoadingUI(false);

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

    setupGui([], net);
    setupFPS();
    detectPoseInRealTime(video, net);
}

// check if access code in browser path
console.log(window.location.href);

// seperate the params
const hashIndex = window.location.href.indexOf('#');
const paramString = window.location.href.substr(hashIndex + 1);

var urlParams = new URLSearchParams(paramString);
if (urlParams.has('access_token')) {
    sessionStorage.setItem('spotify_access_code', urlParams.get('access_token'));
}

function authorize() {
    var redirect = encodeURI(window.location.origin + '/posemusic-web');
    window.location.href = "https://accounts.spotify.com/authorize?client_id=a1dfceece4a24c249303bcdd0a3f865e&redirect_uri=" + redirect + "&scope=user-modify-playback-state%20user-read-playback-state&response_type=token";
}

// check if logged in to spotify
if (sessionStorage.getItem("spotify_access_code") == null) {
    authorize();
}
else {
    var accessToken = sessionStorage.getItem("spotify_access_code");
    spotifyApi.setAccessToken(accessToken);
}

async function updateDeviceList() {
    let deviceResponse;
    try {
        deviceResponse = await spotifyApi.getMyDevices();
    }
    catch (ex) {
        if (ex.status === 401) {
            authorize();
        }
    }
    var devices = deviceResponse.devices;
    var devicesSelect = document.querySelector('#devices');
    devicesSelect.innerHTML = "";
    devices.forEach(device => {
        var optionElement = document.createElement('option');
        optionElement.value = device.id
        optionElement.text = device.name;
        devicesSelect.appendChild(optionElement);
    });
}

function getSelectedDevice() {
    var e = document.getElementById("devices");
    return e.options[e.selectedIndex].value;
}

updateDeviceList();

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
