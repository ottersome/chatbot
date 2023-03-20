--
-- PostgreSQL database dump
--

-- Dumped from database version 13.9
-- Dumped by pg_dump version 14.6 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: users; Type: TABLE; Schema: public; Owner: ottersome
--

CREATE TABLE public.users (
    id integer NOT NULL,
    name character varying(30),
    email character varying(30),
    country_code smallint,
    phone_no smallint,
    created_on timestamp without time zone NOT NULL,
    last_interaction timestamp without time zone
);


ALTER TABLE public.users OWNER TO ottersome;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: ottersome
--

ALTER TABLE public.users ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.users_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: utterances; Type: TABLE; Schema: public; Owner: ottersome
--

CREATE TABLE public.utterances (
    id integer NOT NULL,
    user_id integer,
    text character varying(256) NOT NULL,
    "timestamp" timestamp without time zone NOT NULL
);


ALTER TABLE public.utterances OWNER TO ottersome;

--
-- Name: utterances_id_seq; Type: SEQUENCE; Schema: public; Owner: ottersome
--

ALTER TABLE public.utterances ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.utterances_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: ottersome
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: ottersome
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: utterances fk_userid; Type: FK CONSTRAINT; Schema: public; Owner: ottersome
--

ALTER TABLE ONLY public.utterances
    ADD CONSTRAINT fk_userid FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--

