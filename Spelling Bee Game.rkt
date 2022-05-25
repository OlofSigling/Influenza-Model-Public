;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "htdp-intermediate-lambda-reader.ss" "lang")((modname |Spelling Bee Game|) (read-case-sensitive #t) (teachpacks ((lib "image.rkt" "teachpack" "2htdp"))) (htdp-settings #(#t constructor repeating-decimal #f #t none #f ((lib "image.rkt" "teachpack" "2htdp")) #f)))
;; This project emulates the 7-letter spelling bee game of the NYT, as seen here:
;; https://www.nytimes.com/puzzles/spelling-bee

(require 2htdp/universe)
(require 2htdp/image)
(require 2htdp/batch-io)

(require test-engine/racket-tests)


(define DICTIONARY (read-lines "words.txt"))

;; A Letters is a (NE-List-of 1String)
;; Interpretation: Represents the 7 letters that are available for game of Spelling Bee
;; as a non-empty list of strings. The first item of the list is the letter that is required in
;; all words entered to gain points. 
(define LETTERS-1 (list "l" "o" "e" "b" "w" "m" "s"))
(define LETTERS-2 (list "z" "x" "w" "q" "y" "t" "f")) ;; An essentially impossible set of letters!


(define-struct world [letters word-so-far constructed-words])
;; A World is a (make-world Letters String [Listof-String])
;; Interpretation: A structure that contains information on the letters on the board,
;; the word the user has typed so far, and the words the user has already entered in a list.
(define INIT-WORLD-1 (make-world LETTERS-1 "" (cons "Words so far:" '())))
(define INIT-WORLD-2 (make-world LETTERS-1 "bell" (cons "Words so far:" '())))
(define INIT-WORLD-3 (make-world LETTERS-1 "" (cons "Words so far:" (cons "bell" '()))))

;; world->image : World -> Image
;; Display a World.
(define (world->image w)
  (place-image (score->image (score-calculator (world-constructed-words w))) 40 120
               (overlay (above
                         (text "How many words can you construct?" 25 "black")
                         (rectangle 500 30 "solid" "transparent")
                         (beside/align
                          "top"
                          (above (word->image (world-word-so-far w))
                                 (letters->image (world-letters w)))
                          (cond
                            [(empty? (world-constructed-words w)) (word->image "")]
                            [(cons? (world-constructed-words w))
                             (constructed-word-list->image (world-constructed-words w))])))
                        (empty-scene 700 700))))

;; word->image : String -> Image
;; Displays a word as an image with the color and size that we want.
(define (word->image w)
  (text w 20 "red"))

;; letter->image : 1String -> Image
;; Displays a single letter as an image.
(define BG (empty-scene 500 500))
(define (letter->image s)
  (overlay (text s 60 "blue")
           (circle 35 "outline" "black")))

;; letters->image : Letters -> Image
;; Displays a Letters.
(define (letters->image l)
  (place-image (letter->image (seventh l)) 150 304
               (place-image (letter->image (sixth l)) 150 196
                            (place-image (letter->image (fifth l)) 250 120
                                         (place-image
                                          (letter->image (fourth l)) 350 196
                                          (place-image (letter->image (third l)) 350 304
                                                       (place-image
                                                        (letter->image (second l)) 250 380
                                                        (overlay
                                                         (letter->image
                                                          (first l)) BG))))))))

; score->image: Number -> Image
; Takes the score of the constructed words and outputs it as an image in the game.
(define (score->image score)
  (text (string-append "Score: " (number->string score)) 20 "black"))

;; handler
; constructed-word-list->image: World -> Image
; Takes a world and makes the already constructed words into an image.
(define (constructed-word-list->image list)
  (cond
    [(empty? list) (word->image "")]
    [(cons? list) (above (word->image (first list))
                         (constructed-word-list->image (rest list)))]))

;; available-letter? : Letters String -> Boolean
;; Checks if an entered letter is part of the Letters.
(define (available-letter? l k)
  (ormap (lambda (x) (string=? x k)) l))

;; key-pressed : World KeyEvent -> World
;; Produce a new World in reaction to a key-press.
(define (key-pressed w k)
  (cond
    [(available-letter? (world-letters w) k)
     (make-world (world-letters w) (string-append (world-word-so-far w) k)
                 (world-constructed-words w))]
    [(and (string=? k "\r")
          (in-dictionary? (world-word-so-far w))
          (list-not-duplicate? (world-word-so-far w) (world-constructed-words w))
          (string-contains? (first (world-letters w)) (world-word-so-far w)) (4-letters?
                                                                              (world-word-so-far w)))
     (make-world (world-letters w)
                 ""
                 (append (world-constructed-words w) (cons (world-word-so-far w) '())))]
    [(string=? k "\b") (make-world (world-letters w) (backspace (world-word-so-far w))
                                   (world-constructed-words w))]
    [else w]))


; 4-letters? String -> Boolean
; Checks if a string is atleast 4 letters
(define (4-letters? word)
  (< 3 (string-length word)))

; list-not-duplicate?: String [List-of String] -> Boolean
; Takes a list of constructed words and checks if the word that is currently being entered
;; has already been used
(check-expect (list-not-duplicate? "bell" (list "bell" "mellow" "elbow")) #false)
(check-expect (list-not-duplicate? "elbow" (list "bell" "mellow")) #true)
(define (list-not-duplicate? word los)
  (not (ormap (λ (w) (duplicate? w word)) los)))


; duplicate?: String String -> Boolean
; Checks if a string is already entered in the word game
(define (duplicate? word str)
  (string=? word str))

; in-dictionary?: String -> Boolean
; Takes a dictionary and checks if the word entered is in the dictionary
(define (in-dictionary? word)
  (ormap (λ (w) (string=? w word)) DICTIONARY))
 
; backspace: String -> String
; removes the last character of a word or returns nothing if the string is empty
(define (backspace word)
  (cond
    [(= (string-length word) 0) word]
    [else (substring word 0 (- (string-length word) 1))]))


;; Scoring system: 
;; 1. One point for a four-letter word,
;; 2. An additional point for every additional letter beyond the first four, and
;; 3. An additional seven bonus points for using all seven letters.

; score-calculator: [List-of String] -> Number
; Takes the list of constructed words and calculates the score
(define (score-calculator los)
  (cond
    [(empty? los) 0]
    [(cons? los) (foldr word-score 0 los)]))

; word-score: String -> Number
; Checks how many point an entered word is worth
(define (word-score word starting)
  (+ starting (+ 1
                 (cond
                   [(string=? word "Words so far:") -1]
                   [(<= (string-length word) 4) 0]
                   [(= (string-length word) 5) 1]
                   [(= (string-length word) 6) 2]
                   [(= (string-length word) 7) 10]))))

;; play : World -> World
;;
;; Uses big-bang to play a game of Spelling Bee, given Letters.
(define (play w)
  (big-bang
      w
    (to-draw world->image)
    (on-key key-pressed)))

(play INIT-WORLD-1)
