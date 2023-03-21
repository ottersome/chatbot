/*
 * Starter Project for WhatsApp Echo Bot Tutorial
 *
 * Remix this as the starting point for following the WhatsApp Echo Bot tutorial
 *
 */

"use strict";

//-----------------------------------
// Create Router
//-----------------------------------
const express = require('express')
const router  =  express.Router()
const axios = require("axios").default
const mcache = require('memory-cache')

// Access token for your app
// (copy token from DevX getting started page
// and save it as environment variable into the .env file)
const token = process.env.WHATSAPP_TOKEN;
const { spawn, exec } = require('child_process')
const pgp = require('pg-promise')()


const db_user = process.env.DB_USER
const db_pass = process.env.DB_PASS

const cn = {
    host: 'localhost',
    port: 5432,
    database: 'parrot',
    user: db_user,
    password: db_pass,
    max: 30 // use up to 30 connections

    // "types" - in case you want to set custom type parsers on the pool level
};

class Client{
    //TODO maybe we could add previous history as a parameter
    constructor(num){
        this.history = ""
        //TODO work on the country code
        this.user_id = null
        this.country_code = 0
        this.num =num
        this.pnum_id = -1
        this.name = "User"
        //this.status = ClientStatus.WAITING_BOT
    }
}

// Connect to the database
const db = pgp(cn)

// Imports dependencies and set up http server

//Queue
const queue_class = require('./queue')
const queue = new queue_class()

//Initialize the bot here
let betty = spawn('/home/racc/miniconda3/envs/racc_res/bin/python3.8',
    ['/home/racc/NLP/chatbot/models/gptj/chatbot.py','/home/racc/NLP/chatbot/models/gptj/output/chkpnt-2-1186'])
betty.stdout.setEncoding('utf8')

betty.stdout.on('data',async (data) => {

    //Get the serviced user out of the queue
    let curr_dequeue = queue.dequeue()
    let serviced_user = await retrieve_user(curr_dequeue)
    //Send Response to User
    console.log("Bot responded to user "+serviced_user.name+": " + data)
    //let response = data.toString().replace(/(\r\n|\n|\r)/gm, "");
    let response = data.toString()
    let last_index= response.lastIndexOf('\n')
    response = response.slice(0,last_index)
    send_response(serviced_user.num, response, serviced_user.pnum_id)

    //Save Response
    save_utterance(serviced_user.user_id,serviced_user.num, response, true)

    // Prep next message
    //Get new user
    if(!queue.isEmpty()){
        let newuser = await retrieve_user(queue.peek())
        write_to_betty(newuser.user_id,newuser.num, newuser.history)
    }
    // res.sendStatus(200)
});

function save_utterance(uid, pnum, text, isbot){
    let cur_timestamp = Date.now()
    if(mcache.get(pnum) !=null){
        mcache.get(pnum).history+=text+'|';
    }
    db.result(`INSERT INTO utterances(user_id,text,timestamp,isbot) VALUES($1,$2,to_timestamp(${cur_timestamp}/1000.0),$3) RETURNING text;`,
        [uid,text,isbot]).then(text =>{
            console.log("Saved bot utterance "+text)
        }).catch(error=>{
            console.log("Error when trying to save users text:"+ error)
        })
}

function write_to_betty(txt){
    //Check if we have cached history
    betty.stdin.write(txt+'\r\n')
}

async function create_user_in_db(users_name, users_num, pnum_id){
        let cur_timestamp = Date.now()
        console.log("No user. Creating them")
        //TODO create function to handle country codes
        await db.result(`INSERT INTO users(name,country_code,phone_no,created_on,pnum_id) VALUES($1,$2,$3,to_timestamp(${cur_timestamp}/1000.0),$4) RETURNING id;`,
            [users_name,0,users_num, pnum_id])
        send_response(users_num,'Al usar este bot, usted esta registrando su numero de telefono y nombre en nuestra base de datos. Esto es usado para dos razones principales:\
        Mantener contexto entre conversaciones, y entrenar el bot para tener mejor desempeno. Dado a que el bot esta en desarrollo estas condiciones no son opcionales. Si no esta deacuerdo con esto\
        siempre puede mandar el comando: "cancelar" para remover su informacion de nuestra base de datos. Si le gustaria saber mas de nuestras utilidades escriba el comando "ayuda".',pnum_id)

        console.log("User Created for "+users_name)
}

async function retrieve_user_from_db(users_num){
    //See if the user exists in the database and then pull his history
    try{
        const user = await db.any('SELECT * FROM users WHERE phone_no = $1;',[users_num])
        let cuser =  new Client(-1)
        if (user.length > 0){
            //Pull Out their history
            console.log("Found User. Retrieving them")
            const user_id = user[0].id
            const utterances = await db.any('SELECT text FROM utterances WHERE user_id = $1 and \
id > (SELECT last_utterance_id FROM context_partitions WHERE user_id = $1 ORDER BY last_utterance_id DESC LIMIT 1);',[user_id])
            //TODO filter it so that only < 512 tokens are here
            let history = ""
            for(var i =0;i<utterances.length;i++){
                history+=utterances[i].text +'|'
            }
            console.log("Got User history:\n--------------"+ history+"\n---End of history---")
            cuser.history = history;cuser.user_id = user[0].id
            cuser.pnum_id = user[0].pnum_id
            cuser.num = user[0].phone_no; cuser.name=user[0].name
            //
        }
        return cuser;//Will have -1 phone number in case it was not retrieved
    }catch(e){
        //TODO handle the error
        console.log('We had a problem querying the user out:'+e)
    }
    //Return the Cached User to the request object
    return cuser
}

async function retrieve_user(user_pnum){
    let cached_user = mcache.get(user_pnum) 
    if(cached_user!= null){
        console.log("Retrieving user "+cached_user.name+' from cache')
    }else{
        cached_user = await retrieve_user_from_db(user_pnum)
    }
    //Check for context that is too long
    let len = cached_user.history.length
    while(len > 1800){//TODO make it a bit better
        let first_sep = cached_user.history.indexOf("|")
        cached_user.history.slice(first_sep+1)
        len = cached_user.history.length
    }

    return cached_user
}

function check_is_whatsapp_message(req,res, next){
    if (req.body.object &&
        req.body.entry &&
        req.body.entry[0].changes &&
        req.body.entry[0].changes[0] &&
        req.body.entry[0].changes[0].value.messages &&
        req.body.entry[0].changes[0].value.messages[0]
    ) {
        //Might as well help and set up future values
        req.phone_num_id = req.body.entry[0].changes[0].value.metadata.phone_number_id;
        req.phone_num = req.body.entry[0].changes[0].value.messages[0].from; // extract the phone number from the webhook payload
        req.from_name = req.body.entry[0].changes[0].value.contacts[0].profile.name; // extract the phone number from the webhook payload
        req.msg_body = req.body.entry[0].changes[0].value.messages[0].text.body; // extract the message text from the webhook payload
        next()

    }else{
        console.log("Weird Commands")
        res.sendStatus(200)
        //We quit our logic pipeline here
    }
}

async function construct_user(req,res, next){

    console.log('Received a /webhook')  
    if (req.body.object &&
        req.body.entry &&
        req.body.entry[0].changes &&
        req.body.entry[0].changes[0] &&
        req.body.entry[0].changes[0].value.messages &&
        req.body.entry[0].changes[0].value.messages[0]
    ) {

        //Check if there is a cached user we can just get from emmory
        let cached_user = await retrieve_user(req.phone_num)
        console.log("Prelimanry user "+cached_user.name+" has pnum "+cached_user.num)
        if(cached_user.num == -1){//Then we need to create a new user
            console.log("We have to create new user")
            create_user_in_db(req.from_name,req.phone_num, req.phone_num_id)
            return;
        }
        console.log("Debugging User : "+cached_user.num)

        //Add the Recent Message to the History
        console.log("History for user "+cached_user.name+" till now is:\n"+cached_user.history+'\n---End of History---')
        if(mcache.get(cached_user.num)!=null){
            console.log(`User ${cached_user.name} was chached`)
        }else{
            console.log("Placed user "+cached_user.name+" in the cache")
            mcache.put(cached_user.num, cached_user)
        }

        //Now Send them to the next part of the middleware to interact with the bot
        req.user = cached_user
        console.log("Onto next part of pipeline")
        next()
    }else{
        //TODO handle this elsewhere. Possibly in the middleware before it 
        console.log('Dealing with weird body:')
        console.log('With status : '+req.body.entry[0].changes[0].value.statuses.status)
        res.sendStatus(200)
    }
    
}

//async function store_user_utterance(req,res, next){
    //let cur_timestamp = Date.now()
    //db.result(`INSERT INTO utterances(user_id,text,timestamp,isbot) VALUES($1,$2,to_timestamp(${cur_timestamp}/1000.0),false) RETURNING text;`,
        //[req.user.user_id,req.msg_body]).then(data =>{
            ////console.log("Saved utterance ")
        //}).catch(error=>{
            //console.log("Error when trying to save users text "+error)
        //})
    //console.log("Onto final part of pipeline.")
    //next()
//}

async function clean_context(req){
    //Clear the cached history of the user
    req.user.history = ""//Cleared
    //Now Insert into database a new contet partition
    //By this point we should have user id
    let user_id = req.user.user_id
    const res = await db.result(`INSERT INTO context_partitions(last_utterance_id,user_id)\
        VALUES((SELECT id FROM utterances WHERE user_id = `+user_id+` ORDER BY id DESC LIMIT 1), `+user_id+`);`);
    if(res< 0){//TODO do somethign

    }
    console.log('Cleared Context')

}
function parse_commands(req,res, next){
    switch(req.msg_body){
        case "ayuda":
            send_response(req.phone_num,
                "Esto es mensaje de ayuda",req.phone_num_id)
            break
        case "limpiar":
            console.log("Cleaning Context")
            clean_context(req)
            break
        case "regenerar":
            console.log("Regenerating Response")
            break
        case "rate":
            console.log("Rating response")
        default:
            next()
    }

}

//router.post("/", construct_user)
//TODO add users message to the database

// Accepts POST requests at /webhook endpoint
//router.post("/webhook", (req, res) => {
router.post("/", check_is_whatsapp_message,construct_user, parse_commands,  (req, res) => {
    console.log('Woopidy toot. We got  a request:')
    // Parse the request body from the POST
    let client = req.user
    // Check the Incoming webhook message
    console.log('Received a /webhook')  
    if (client) {
        console.log("This request has a body")
        console.log("It has the characterisitcs of a message")
        console.log("From : "+req.body.entry[0].changes[0].value.contacts[0].profile.name)
        console.log("Message body:"+req.body.entry[0].changes[0].value.messages[0].text.body)
        console.log('Debugging Done')
        res.sendStatus(200)

        if (queue.has(req.phone_num)){
            console.log("Found client in queue")
            send_response(req.phone_num,'Your previous response is being processed please wait patiently.',req.phone_num_id)
        }else{//Push Client to the queuej
            console.log("Enqueueing client  "+req.from_name+" with pnum :"+client.num)
            queue.enqueue(client.num)
            if(queue.size() == 1){
                console.log("History b4:"+ client.history)
                save_utterance(client.user_id,client.num,req.msg_body,false)
                console.log("History 4fter:"+ client.history)
                let to_betty = client.history
                console.log("Writing to betty: "+to_betty+'\n')
                write_to_betty(to_betty)
            }else{
                console.log("Nothing done cuz queue size is :"+queue.size())

            }
            //TODO remove below when already replaced by other logic
        }
    } 
});

function send_response(to_pn, text,pn_id){
    console.log(`Trying to send response to phone number ${to_pn} with id ${pn_id} and final text: ${text}`)
            axios({
                method: "post", // Required, HTTP method, a string, e.g. POST, GET
                url:
                "https://graph.facebook.com/v15.0/" +
                pn_id+`/messages?access_token=${token}`,//+
                // "/messages?access_token=" +
                // token,
                data: {
                    messaging_product: "whatsapp",
                    to: to_pn,
                    text: { body: text},
                    type : "text",
                    // template : { "name": "hello_world", "language": { "code": "en_US" } }
                },
                headers: {
                    'Authorization': `Bearer ${token}`,
                    "Content-Type": "application/json" },
            }).then(function(res){
                console.log('Things are fine here')
                // console.log(res.data)
                // console.log(res.status)
                // res.sendStatus(200)
                // console.log(res.statusText) // console.log(res.headers) // console.log(res.config) res.sendStatus(200);
            }).catch(function(error){
                console.log(error);
                console.log('Oh no something broke here')
                return;
            });
}
module.exports = router
