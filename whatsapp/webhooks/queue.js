class Queue {
    constructor() {
        this.items = [];
    }
    
    // add element to the queue
    enqueue(element) {
        return this.items.push(element);
    }
    
    // remove element from the queue
    dequeue() {
        if(this.items.length > 0) {
            return this.items.shift();
        }
    }
   
    // the size of the queue
    size(){
        return this.items.length;
    }
    has(num){
        //Just do linear search as there is no ordering to enforce
        //TODO: Fix this hack of using a string instead of the actual number
        for(var i = 0;i < this.size();i++)
            if (this.items[i].num == num)
                return true
        return false
    }

    
    // view the last element
    peek() {
        return this.items[this.items.length - 1];
    }
    
    // check if the queue is empty
    isEmpty(){
       return this.items.length == 0;
    }
 
    // empty the queue
    clear(){
        this.items = [];
    }
}
module.exports = Queue
