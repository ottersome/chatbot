const express = require('express')
const request = require("request"),
    body_parser = require("body-parser"),
    https = require('https'),
    fs = require('fs')
const app = express()

var bole = require('bole')
bole.output({level: 'debug', stream: process.stdout})
var log = bole('server')
log.info('Server Starting '+new Date().toLocaleString('en-US', {timeZone: 'Asia/Taipei'}))

//----------------------------------------
//Set some of the middleweb
//----------------------------------------
//app.use(require('./website/router'))
app.use(body_parser.json())
app.use('/',(req,res, next) => {
    console.log('New Request'); next();})
//app.use('/webhook',require('./chatbot'))
app.post("/webhook", (req,res, next) => {console.log("Poop"); next();})
app.use("/webhook",require('./chatbot'))

//----------------------------------------
// Set up HTTPS
//----------------------------------------
const cert = fs.readFileSync(process.env.CERT);
const ca = fs.readFileSync(process.env.CA);
const key = fs.readFileSync(process.env.KEY);
//For HTTPS Stuff
var https_config = {
    key: key,
    ca: ca,
    cert: cert
}

var listener = https.createServer(https_config,app)
    .listen(process.env.PORT || 1337, () => console.log(`webhook is listening ${listener.address().port}`));
