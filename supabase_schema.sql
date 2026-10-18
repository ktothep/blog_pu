-- ============================================================
-- TailorMade schema migration (run in Supabase → SQL Editor)
-- Adds: richer resume history columns, targeting profile,
--       applications tracker, and artifacts (cover letters,
--       interview prep).
-- Safe to re-run (idempotent).
-- ============================================================

alter table public.user_activity
    add column if not exists ats_score_tailored numeric,
    add column if not exists original_resume   text,
    add column if not exists markdown_resume   text,
    add column if not exists job_description   text,
    add column if not exists style             text default 'standard';

create table if not exists public.user_profile (
    user_id            bigint primary key references public.users(id) on delete cascade,
    target_role        text,
    years_experience   text,
    industry           text,
    locations          text,
    skills_emphasis    text,
    salary_expectations text
);

create table if not exists public.applications (
    id         bigserial primary key,
    user_id    bigint not null references public.users(id) on delete cascade,
    title      text not null,
    company    text,
    job_url    text,
    status     text not null default 'saved',
    notes      text,
    applied_at timestamptz not null default now()
);

create table if not exists public.artifacts (
    id          bigserial primary key,
    user_id     bigint not null references public.users(id) on delete cascade,
    activity_id bigint references public.user_activity(id) on delete set null,
    kind        text not null,            -- 'cover_letter' | 'interview_prep'
    content     text not null,
    created_at  timestamptz not null default now()
);
