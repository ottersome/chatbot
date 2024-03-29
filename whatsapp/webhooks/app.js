/*
 * Starter Project for WhatsApp Echo Bot Tutorial
 *
 * Remix this as the starting point for following the WhatsApp Echo Bot tutorial
 *
 */

"use strict";

// Access token for your app
// (copy token from DevX getting started page
// and save it as environment variable into the .env file)
const token = process.env.WHATSAPP_TOKEN;

// Imports dependencies and set up http server
const request = require("request"),
  express = require("express"),
  body_parser = require("body-parser"),
  axios = require("axios").default,
  https = require('https'),
  fs = require('fs'),
  app = express().use(body_parser.json()); // creates express http server

//Read SSL information
const cert = fs.readFileSync(process.env.CERT);
const ca = fs.readFileSync(process.env.CA);
const key = fs.readFileSync(process.env.KEY);

// Sets server port and logs message on success
var https_config = {
    key: key,
    ca: ca,
    cert: cert
}
var listener = https.createServer(https_config,app)
    .listen(process.env.PORT || 1337, () => console.log(`webhook is listening ${listener.address().port}`));

app.get("/", function(request, response){
    response.send('Hi there </br> what are you doing here .-."');
});

// Accepts POST requests at /webhook endpoint
app.post("/webhook", (req, res) => {
  // Parse the request body from the POST
  let body = req.body;

  // Check the Incoming webhook message

  console.log('Received a /webhook')  
  // console.log(JSON.stringify(req.body, null, 2));

  // info on WhatsApp text message payload: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples#text-messages
  if (req.body.object) {
      console.log("This request has a body")
    if (
      req.body.entry &&
      req.body.entry[0].changes &&
      req.body.entry[0].changes[0] &&
      req.body.entry[0].changes[0].value.messages &&
      req.body.entry[0].changes[0].value.messages[0]
    ) {
      console.log("It has the characteristaics of a message")
      let phone_number_id =
        req.body.entry[0].changes[0].value.metadata.phone_number_id;
      let from = req.body.entry[0].changes[0].value.messages[0].from; // extract the phone number from the webhook payload
      let from_name = req.body.entry[0].changes[0].value.contacts[0].profile.name; // extract the phone number from the webhook payload
      let msg_body = req.body.entry[0].changes[0].value.messages[0].text.body; // extract the message text from the webhook payload
      console.log(`and the message body is ${msg_body}`)
      console.log(`your token is : ${token}`)
      axios({
        method: "post", // Required, HTTP method, a string, e.g. POST, GET
        url:
          "https://graph.facebook.com/v15.0/" +
          phone_number_id+`/messages?access_token=${token}`,//+
          // "/messages?access_token=" +
          // token,
        data: {
          messaging_product: "whatsapp",
          to: from,
          text: { body: `Hey ${from_name}. We are currently under maintenance. Please visit us later` },
          type : "text",
          // template : { "name": "hello_world", "language": { "code": "en_US" } }
        },
        headers: {
            'Authorization': `Bearer ${token}`,
            "Content-Type": "application/json" },
      }).then(function(res){
          console.log(res.data)
          console.log(res.status)
          // console.log(res.statusText)
          // console.log(res.headers)
          // console.log(res.config)

      }).catch(function(error){
          console.log(error);
          res.sendStatus(500);
          return;
      });
    }
    res.sendStatus(200);
  } else {
    // Return a '404 Not Found' if event is not from a WhatsApp API
    res.sendStatus(404);
  }
});

// Accepts GET requests at the /webhook endpoint. You need this URL to setup webhook initially.
// info on verification request payload: https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests 
app.get("/webhook", (req, res) => {
  /**
   * UPDATE YOUR VERIFY TOKEN
   *This will be the Verify Token value when you set up webhook
  **/
  const verify_token = process.env.VERIFY_TOKEN;
  console.log(`Getting the verification token`)

  // Parse params from the webhook verification request
  let mode = req.query["hub.mode"];
  let token = req.query["hub.verify_token"];
  let challenge = req.query["hub.challenge"];
  console.log(`Token is ${token}`)

  // Check if a token and mode were sent
  if (mode && token) {
    // Check the mode and token sent are correct
    if (mode === "subscribe" && token === verify_token) {
      // Respond with 200 OK and challenge token from the request
      console.log("WEBHOOK_VERIFIED");
      res.status(200).send(challenge);
    } else {
      // Responds with '403 Forbidden' if verify tokens do not match
      res.sendStatus(403);
    }
  }else res.sendStatus(403)
});

