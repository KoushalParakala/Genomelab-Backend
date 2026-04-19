-- Supabase Schema for AI Mutation Platform Database Migration

-- Ensure UUID extension is available
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Profiles Table (extends standard Supabase Auth users)
CREATE TABLE public.profiles (
    id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- RLS for profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view their own profile" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update their own profile" ON public.profiles FOR UPDATE USING (auth.uid() = id);

-- 2. Sequence Records Table (baselines)
CREATE TABLE public.sequence_records (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    sequence_data TEXT NOT NULL,
    length INTEGER NOT NULL,
    gc_content FLOAT,
    is_valid BOOLEAN DEFAULT TRUE,
    translation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- RLS for sequences
ALTER TABLE public.sequence_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view their own sequences" ON public.sequence_records FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own sequences" ON public.sequence_records FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can delete their own sequences" ON public.sequence_records FOR DELETE USING (auth.uid() = user_id);

-- 3. Experiments Table
CREATE TABLE public.experiments (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- RLS for experiments
ALTER TABLE public.experiments ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view their own experiments" ON public.experiments FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own experiments" ON public.experiments FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own experiments" ON public.experiments FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own experiments" ON public.experiments FOR DELETE USING (auth.uid() = user_id);

-- 4. Mutation Logs Table
CREATE TABLE public.mutation_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    experiment_id UUID REFERENCES public.experiments(id) ON DELETE SET NULL,
    baseline_id UUID REFERENCES public.sequence_records(id) ON DELETE CASCADE,
    mutated_sequence_data TEXT NOT NULL,
    mutation_type TEXT NOT NULL,
    position INTEGER NOT NULL,
    new_nucleotide TEXT,
    variant_annotation TEXT,
    stability_score FLOAT,
    functional_risk TEXT,
    aggregation_risk FLOAT,
    -- AI Pipeline Fields
    embedding_distance FLOAT,
    structural_rmsd FLOAT,
    classifier_confidence FLOAT,
    biological_narrative TEXT,
    esm_embedding JSONB,
    structure_pdb TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- RLS for mutation_logs
ALTER TABLE public.mutation_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view their own mutation logs" ON public.mutation_logs FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own mutation logs" ON public.mutation_logs FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can delete their own mutation logs" ON public.mutation_logs FOR DELETE USING (auth.uid() = user_id);

-- 5. Shared Results (public links)
CREATE TABLE public.shared_results (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    experiment_id UUID REFERENCES public.experiments(id) ON DELETE CASCADE NOT NULL,
    shared_by UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    token TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- RLS for shared_results (public can read if they have token, owner can manage)
ALTER TABLE public.shared_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Owner can manage shared results" ON public.shared_results FOR ALL USING (auth.uid() = shared_by);
CREATE POLICY "Anyone can view shared result with token" ON public.shared_results FOR SELECT USING (true);

-- Triggers

-- Trigger to create profile after Auth signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name)
  VALUES (new.id, new.email, new.raw_user_meta_data->>'full_name');
  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- Trigger to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_profiles_modtime BEFORE UPDATE ON public.profiles FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_experiments_modtime BEFORE UPDATE ON public.experiments FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
